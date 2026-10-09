"""Causal multi-turn inference for QueryStream++."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoProcessor

from querystream_plus.artifacts import resolve_artifacts, validate_inference_artifacts
from querystream_plus.memory.manager import ComplementaryTokenMemory
from querystream_plus.memory.features import VisualEvidence, encode_visual_evidence
from querystream_plus.memory.packing import (
    QueryStreamPlusForConditionalGeneration,
    pack_visual_context,
)
from querystream_plus.inference.vision import (
    _first_real_device,
    _add_semantic_encoder_args,
    _load_semantic_encoder,
    _semantic_encoder_spec,
    _normalize_device_map,
    load_video_frames,
    resolve_attention_backend,
)
from querystream_plus.modeling.router import RouterConfig, build_router, select_routed_tokens


def unwrap_backbone(model):
    """Return the QueryStream++ backbone under an optional PEFT wrapper."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _load_lora_adapter(model, adapter_path: Optional[str]):
    if adapter_path is None:
        return model
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "LoRA inference requires PEFT. Install the project requirements or run `pip install peft==0.17.1`."
        ) from exc
    load_kwargs = {"is_trainable": False}
    # Keep the base model's device placement.
    device_map = getattr(model, "hf_device_map", None)
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    return PeftModel.from_pretrained(model, adapter_path, **load_kwargs)


def _local_path(path: str) -> str:
    return path[7:] if path.startswith("file://") else path


def _load_session(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        session = json.load(handle)
    if not isinstance(session.get("video"), str):
        raise ValueError("Session JSON must contain a string 'video' path.")
    turns = session.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("Session JSON must contain a non-empty 'turns' list.")
    previous_end = 0.0
    for index, turn in enumerate(turns):
        if not isinstance(turn.get("query"), str) or not turn["query"].strip():
            raise ValueError(f"turns[{index}].query must be non-empty.")
        visible_end = float(turn.get("visible_end", -1.0))
        if visible_end <= previous_end:
            raise ValueError("visible_end must increase strictly across QueryStream++ turns.")
        previous_end = visible_end
    return session


def _resolve_video_path(session_json: str, video: str) -> str:
    video = _local_path(video)
    if os.path.isabs(video):
        return video
    return str((Path(session_json).resolve().parent / video).resolve())


def _build_messages(history: Sequence[Dict[str, str]], frames, query: str) -> list:
    messages = []
    for turn in history:
        messages.append({"role": "user", "content": [{"type": "text", "text": turn["query"]}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": turn["answer"]}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": query},
            ],
        }
    )
    return messages


def _visible_frame_boundary(timestamps: Sequence[float], seconds: float) -> int:
    return bisect.bisect_right(timestamps, seconds + 1e-9)


def _prepare_chunk(frames: Sequence, start: int, end: int) -> List:
    chunk = list(frames[start:end])
    if not chunk:
        raise ValueError("Each QueryStream++ turn must expose at least one new sampled frame pair.")
    while len(chunk) < 2:
        chunk.append(chunk[-1].copy())
    if len(chunk) % 2 == 1:
        chunk.append(chunk[-1].copy())
    return chunk


def inject_visual_embeddings(model, input_ids: torch.Tensor, visual_values: torch.Tensor) -> torch.Tensor:
    model = unwrap_backbone(model)
    embeds = model.model.embed_tokens(input_ids)
    mask = input_ids == model.config.video_token_id
    if int(mask.sum().item()) != visual_values.shape[0]:
        raise ValueError("Backbone visual embedding count does not match the prompt video span.")
    return embeds.masked_scatter(
        mask.unsqueeze(-1).expand_as(embeds),
        visual_values.to(device=embeds.device, dtype=embeds.dtype),
    )


def extract_query_context(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    video_token_id: int,
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    mask = input_ids != video_token_id
    selected = inputs_embeds[0, mask[0]].unsqueeze(0)
    return selected, torch.ones(selected.shape[:2], dtype=torch.bool, device=selected.device)


def route_current_tokens(
    features: VisualEvidence,
    current_visual_values: torch.Tensor,
    input_ids: torch.Tensor,
    full_inputs_embeds: torch.Tensor,
    video_token_id: int,
    active_budget: int,
    router,
) -> Tuple[torch.Tensor, torch.BoolTensor, str]:
    device = current_visual_values.device
    dtype = current_visual_values.dtype
    visual_hidden = current_visual_values.unsqueeze(0)
    text_context, text_attention_mask = extract_query_context(
        full_inputs_embeds, input_ids, video_token_id
    )
    with torch.no_grad():
        refined, _ = router(
            visual_hidden=visual_hidden,
            text_context=text_context,
            prior_score=features.prior_score.unsqueeze(0).to(device=device, dtype=dtype),
            prior_features=features.prior_features.unsqueeze(0).to(device=device, dtype=dtype),
            residual_scale=1.0,
            semantic_hidden=features.semantic_keys.unsqueeze(0).to(device=device, dtype=dtype),
            text_attention_mask=text_attention_mask,
            grid_shape=(
                len(features.metadata) // (features.qwen_grid_hw[0] * features.qwen_grid_hw[1]),
                features.qwen_grid_hw[0],
                features.qwen_grid_hw[1],
            ),
        )
    scores = refined[0].detach()
    active = select_routed_tokens(
        visual_scores=scores,
        active_budget=active_budget,
    )
    return scores, active, "qast+topk"


def load_query_router(
    path: str,
    model,
    device: torch.device,
    dtype: torch.dtype,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Router checkpoint must contain a state dictionary: {path}")
    config = RouterConfig.from_dict(
        checkpoint.get("router_config"),
        hidden_size=model.config.hidden_size,
        semantic_size=1024,
    )
    if config.semantic_size != 1024:
        raise ValueError("Router checkpoint is not compatible with SigLIP2-L/16-256.")
    router = build_router(config).to(device=device, dtype=dtype)
    state = checkpoint.get("router", checkpoint)
    router.load_state_dict(state, strict=True)
    return router.eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QueryStream++ dual-path memory multi-turn inference.")
    parser.add_argument("--session-json", required=True)
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="QueryStreamPlus-7B directory (default: models/QueryStreamPlus-7B).",
    )
    parser.add_argument("--router-checkpoint", default=None)
    parser.add_argument(
        "--lora-adapter",
        default=None,
        help="PEFT adapter directory (default: <checkpoint-path>/adapter).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="flash_attention_2",
        help="Attention backend (default: flash_attention_2).",
    )
    parser.add_argument("--torch-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=1016)
    parser.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    _add_semantic_encoder_args(parser)
    parser.add_argument("--semantic-device", default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--novelty-weight", type=float, default=1.0)
    parser.add_argument(
        "--active-budget",
        type=int,
        default=0,
        help="Fixed current-token budget; 0 derives it from --active-keep-rate.",
    )
    parser.add_argument("--active-keep-rate", type=float, default=0.75)
    parser.add_argument("--qim-capacity", type=int, default=2048)
    parser.add_argument("--im-capacity", type=int, default=3072)
    parser.add_argument("--qim-read-budget", type=int, default=1024)
    parser.add_argument("--im-read-budget", type=int, default=1536)
    parser.add_argument("--active-buffer-capacity", type=int, default=2048)
    parser.add_argument("--im-episode-budget", type=int, default=128)
    parser.add_argument("--memory-state-in", default=None)
    parser.add_argument("--memory-state-out", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _dtype_argument(name: str):
    if name == "auto":
        return "auto"
    return getattr(torch, name)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    artifacts = resolve_artifacts(
        model_path=args.checkpoint_path,
        router_checkpoint=args.router_checkpoint,
        lora_adapter=args.lora_adapter,
        siglip2_checkpoint=args.siglip2_checkpoint,
    )
    validate_inference_artifacts(artifacts)
    args.checkpoint_path = str(artifacts.model)
    args.router_checkpoint = str(artifacts.router)
    args.lora_adapter = str(artifacts.adapter)
    args.siglip2_checkpoint = str(artifacts.siglip2)
    session = _load_session(args.session_json)
    video_path = _resolve_video_path(args.session_json, session["video"])
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    device_map = {"": "cpu"} if args.cpu_only else _normalize_device_map(args.device_map)
    model_kwargs = {"torch_dtype": _dtype_argument(args.torch_dtype), "device_map": device_map}
    if device_map is None:
        model_kwargs.pop("device_map")
    requested_attention = args.attn_implementation
    attention_device = "cpu" if args.cpu_only else args.device
    model_kwargs["attn_implementation"] = resolve_attention_backend(
        requested_attention, attention_device
    )
    print(f"Loading QueryStream++ from {args.checkpoint_path}...", file=sys.stderr)
    model = QueryStreamPlusForConditionalGeneration.from_pretrained(
        args.checkpoint_path, **model_kwargs
    )
    if device_map is None:
        model = model.to(torch.device(args.device))
    model = _load_lora_adapter(model, args.lora_adapter).eval()
    qwen = unwrap_backbone(model)
    processor = AutoProcessor.from_pretrained(args.checkpoint_path)
    target_device = _first_real_device(model)
    model_dtype = next(parameter.dtype for parameter in model.parameters() if parameter.is_floating_point())

    semantic_device_name = args.semantic_device or ("cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda")
    semantic_device = torch.device(semantic_device_name)
    semantic_model, semantic_checkpoint = _semantic_encoder_spec(args)
    semantic_encoder, semantic_preprocess, semantic_processor = _load_semantic_encoder(
        semantic_model, semantic_checkpoint, semantic_device
    )
    router = load_query_router(
        args.router_checkpoint,
        model,
        target_device,
        model_dtype,
    )

    restored_dsh = None
    restored_boundary = 0
    restored_history: List[Dict[str, str]] = []
    restored_turn_id = 0
    restored_temporal_offset = 0
    if args.memory_state_in:
        saved_state = torch.load(args.memory_state_in, map_location="cpu")
        if "manager" in saved_state:
            manager = ComplementaryTokenMemory.from_state_dict(saved_state["manager"])
            restored_dsh = saved_state.get("dsh_state")
            restored_boundary = int(saved_state.get("previous_boundary", 0))
            restored_history = list(saved_state.get("history", []))
            restored_turn_id = int(saved_state.get("next_turn_id", len(restored_history)))
            restored_temporal_offset = int(
                saved_state.get("temporal_offset", restored_boundary // 2)
            )
            saved_video = saved_state.get("video")
            if saved_video and os.path.abspath(saved_video) != os.path.abspath(video_path):
                raise ValueError("The restored memory session belongs to a different video.")
        else:
            raise ValueError("Invalid QueryStream++ memory state.")
    else:
        manager = ComplementaryTokenMemory(
            qim_capacity=args.qim_capacity,
            im_capacity=args.im_capacity,
            active_buffer_capacity=args.active_buffer_capacity,
            im_episode_budget=args.im_episode_budget,
        )

    all_frames, frame_timestamps = load_video_frames(
        video_path,
        fps=args.fps,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        return_timestamps=True,
    )
    history: List[Dict[str, str]] = restored_history
    outputs = []
    previous_boundary = restored_boundary
    temporal_offset = restored_temporal_offset
    dsh_state = restored_dsh
    for local_turn_id, turn in enumerate(session["turns"]):
        turn_id = restored_turn_id + local_turn_id
        visible_end = float(turn["visible_end"])
        boundary = _visible_frame_boundary(frame_timestamps, visible_end)
        if boundary <= previous_boundary:
            raise ValueError(
                "After FPS sampling, each turn must add a new frame; increase visible_end spacing or FPS."
            )
        chunk = _prepare_chunk(all_frames, previous_boundary, boundary)
        query = turn["query"]
        messages = _build_messages(history, chunk, query)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[prompt], images=None, videos=[chunk], padding=True, return_tensors="pt"
        ).to(target_device)
        visual_values, visual_type = qwen.embedding_visual_info(**inputs)
        if visual_type != "video":
            raise ValueError("QueryStream++ expected video embeddings from its backbone.")
        visual_values = visual_values.to(device=target_device, dtype=model_dtype)

        features = encode_visual_evidence(
            video_frames=chunk,
            video_grid_thw=inputs.video_grid_thw.detach().cpu(),
            query=query,
            semantic_encoder=semantic_encoder,
            semantic_preprocess=semantic_preprocess,
            semantic_processor=semantic_processor,
            semantic_device=semantic_device,
            temporal_alpha=args.alpha,
            novelty_weight=args.novelty_weight,
            fps=args.fps,
            global_temporal_offset=temporal_offset,
            initial_dsh=dsh_state,
        )
        dsh_state = features.temporal_state
        temporal_offset += int(inputs.video_grid_thw[0, 0])
        if visual_values.shape[0] != features.semantic_keys.shape[0]:
            raise ValueError(
                f"Aligned feature count {features.semantic_keys.shape[0]} does not match "
                f"backbone visual count {visual_values.shape[0]}."
            )

        full_inputs_embeds = inject_visual_embeddings(model, inputs.input_ids, visual_values)
        active_budget = args.active_budget or max(
            1, int(round(visual_values.shape[0] * args.active_keep_rate))
        )
        router_scores, active_mask, selector = route_current_tokens(
            features=features,
            current_visual_values=visual_values,
            input_ids=inputs.input_ids,
            full_inputs_embeds=full_inputs_embeds,
            video_token_id=qwen.config.video_token_id,
            active_budget=active_budget,
            router=router,
        )

        recalled = manager.retrieve(
            query_embedding=features.query_embedding,
            qim_budget=args.qim_read_budget,
            im_budget=args.im_read_budget,
        )
        manager.observe(
            semantic_keys=features.semantic_keys,
            qwen_values=visual_values.detach().cpu(),
            novelty=features.novelty,
            router_scores=router_scores,
            active_mask=active_mask,
            token_metadata=features.metadata,
            turn_id=turn_id,
        )
        packed = pack_visual_context(
            model=qwen,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            current_visual_values=visual_values,
            current_keep_mask=active_mask,
            current_metadata=features.metadata,
            recalled_slots=recalled.slots,
        )
        qwen.set_memory_prefill_positions(packed.position_ids, packed.sequence_length)
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
        }
        if args.do_sample:
            generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
        generated_ids = model.generate(
            input_ids=packed.input_ids,
            inputs_embeds=packed.inputs_embeds,
            attention_mask=packed.attention_mask,
            position_ids=packed.position_ids,
            use_cache=True,
            **generation_kwargs,
        )
        generated_only = (
            generated_ids[:, packed.sequence_length :]
            if generated_ids.shape[1] > packed.sequence_length
            else generated_ids
        )
        answer = processor.batch_decode(
            generated_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        manager.consolidate_turn(features.query_embedding, turn_id)
        memory_stats = manager.stats()
        history.append({"query": query, "answer": answer})
        turn_output = {
            "turn_id": turn_id,
            "visible_end": visible_end,
            "query": query,
            "answer": answer,
            "selector": selector,
            "chunk_frames": len(chunk),
            "current_visual_tokens": int(active_mask.sum().item()),
            "recalled_qim_tokens": len(recalled.qim_slots),
            "recalled_im_tokens": len(recalled.im_slots),
            "packed_visual_tokens": len(packed.visual_sources),
            "memory": memory_stats,
        }
        outputs.append(turn_output)
        print(f"[{visible_end:.1f}s] {answer}", flush=True)
        if args.verbose:
            print(json.dumps(turn_output, ensure_ascii=False), file=sys.stderr)
        previous_boundary = boundary

    model.clear_memory_prefill_positions()
    if args.memory_state_out:
        torch.save(
            {
                "video": os.path.abspath(video_path),
                "router_checkpoint": args.router_checkpoint,
                "manager": manager.state_dict(),
                "dsh_state": dsh_state,
                "previous_boundary": previous_boundary,
                "temporal_offset": temporal_offset,
                "history": history,
                "next_turn_id": restored_turn_id + len(session["turns"]),
            },
            args.memory_state_out,
        )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "video": os.path.abspath(video_path),
            "router_checkpoint": args.router_checkpoint,
            "turns": outputs,
            "final_memory": manager.stats(),
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
