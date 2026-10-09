"""Stateful benchmark inference for QueryStream++."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoProcessor

from querystream_plus.artifacts import resolve_artifacts, validate_inference_artifacts
from querystream_plus.memory.manager import ComplementaryTokenMemory
from querystream_plus.inference.pipeline import (
    route_current_tokens,
    unwrap_backbone,
    _build_messages,
    _first_real_device,
    _load_lora_adapter,
    load_query_router,
    inject_visual_embeddings,
)
from querystream_plus.memory.features import (
    build_visual_evidence,
    encode_aligned_semantic_tokens,
)
from querystream_plus.memory.packing import (
    QueryStreamPlusForConditionalGeneration,
    pack_visual_context,
)
from querystream_plus.inference.vision import (
    _load_semantic_encoder,
    _semantic_encoder_spec,
    load_video_frames,
    resolve_attention_backend,
)
from querystream_plus.inference.video_cache import VideoFrameCache
from querystream_plus.inference.feature_cache import UniversalVisualFeatureCache


@dataclass
class QueryStreamPlusConfig:
    checkpoint_path: Optional[str] = None
    router_checkpoint: Optional[str] = None
    lora_adapter: Optional[str] = None
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    siglip2_model: str = "google/siglip2-large-patch16-256"
    siglip2_checkpoint: Optional[str] = None
    semantic_device: str = "cuda"
    fps: float = 1.0
    min_frames: int = 4
    max_frames: int = 1016
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 256 * 28 * 28
    stream_chunk_frames: int = 16
    active_budget: int = 0
    active_keep_rate: float = 0.75
    qim_capacity: int = 2048
    im_capacity: int = 3072
    active_buffer_capacity: int = 2048
    im_episode_budget: int = 128
    qim_read_budget: int = 1024
    im_read_budget: int = 1536
    temporal_alpha: float = 0.1
    novelty_weight: float = 1.0
    max_new_tokens: int = 128
    visual_cache_dir: Optional[str] = None
    visual_cache_mode: str = "readwrite"


def _dtype(name: str):
    return getattr(torch, name)


class QueryStreamPlusEngine:
    """Stateful evaluator for one video session."""

    def __init__(self, config: QueryStreamPlusConfig):
        artifacts = resolve_artifacts(
            model_path=config.checkpoint_path,
            router_checkpoint=config.router_checkpoint,
            lora_adapter=config.lora_adapter,
            siglip2_checkpoint=config.siglip2_checkpoint,
        )
        validate_inference_artifacts(artifacts)
        config.checkpoint_path = str(artifacts.model)
        config.router_checkpoint = str(artifacts.router)
        config.lora_adapter = str(artifacts.adapter)
        config.siglip2_checkpoint = str(artifacts.siglip2)
        self.config = config
        if config.stream_chunk_frames < 2:
            raise ValueError("stream_chunk_frames must be at least two.")
        if not 0.0 < config.active_keep_rate <= 1.0:
            raise ValueError("active_keep_rate must be in (0, 1].")
        self.chunk_frames = config.stream_chunk_frames - config.stream_chunk_frames % 2
        self.attn_implementation = resolve_attention_backend(
            config.attn_implementation, config.device
        )
        model_kwargs = {
            "torch_dtype": _dtype(config.torch_dtype),
            "attn_implementation": self.attn_implementation,
        }
        self.model = QueryStreamPlusForConditionalGeneration.from_pretrained(
            config.checkpoint_path, **model_kwargs
        ).to(torch.device(config.device))
        self.model = _load_lora_adapter(self.model, config.lora_adapter).eval()
        self.qwen = unwrap_backbone(self.model)
        self.processor = AutoProcessor.from_pretrained(config.checkpoint_path)
        self.device = _first_real_device(self.model)
        self.model_dtype = next(
            parameter.dtype for parameter in self.model.parameters() if parameter.is_floating_point()
        )

        semantic_args = type("SemanticArgs", (), {})()
        semantic_args.siglip2_model = config.siglip2_model
        semantic_args.siglip2_checkpoint = config.siglip2_checkpoint
        semantic_model, semantic_checkpoint = _semantic_encoder_spec(semantic_args)
        self.semantic_device = torch.device(config.semantic_device)
        self.semantic_encoder, self.semantic_preprocess, self.semantic_tokenizer = _load_semantic_encoder(
            semantic_model,
            semantic_checkpoint,
            self.semantic_device,
        )
        self.router = load_query_router(
            config.router_checkpoint,
            self.model,
            self.device,
            self.model_dtype,
        )
        self.frame_cache = None
        self.feature_cache = None
        if config.visual_cache_dir:
            signature = {
                "qwen_base": str(config.checkpoint_path),
                "qwen_hidden_size": int(self.qwen.config.hidden_size),
                "qwen_vision": self.qwen.config.vision_config.to_dict(),
                "semantic_checkpoint": str(semantic_checkpoint or semantic_model),
            }
            self.frame_cache = VideoFrameCache(config.visual_cache_dir, config.visual_cache_mode)
            self.feature_cache = UniversalVisualFeatureCache(
                config.visual_cache_dir, signature, config.visual_cache_mode
            )
        self._stage_timing = False
        self._timing_sample_id = ""
        self._timing_chunk = 0
        self.reset()

    def enable_stage_timing(self, sample_id: str) -> None:
        self._stage_timing = True
        self._timing_sample_id = str(sample_id)
        self._timing_chunk = 0
        print(
            f"[timing][sample={self._timing_sample_id}] enabled for first inferred sample",
            file=sys.stderr,
            flush=True,
        )

    def disable_stage_timing(self) -> None:
        self._stage_timing = False

    def _timing_sync(self) -> None:
        if not self._stage_timing or not torch.cuda.is_available():
            return
        for value in {str(self.device), str(self.semantic_device)}:
            device = torch.device(value)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    def _timing_start(self):
        if not self._stage_timing:
            return None
        self._timing_sync()
        return time.perf_counter()

    def _timing_print(self, stage: str, started) -> None:
        if started is None:
            return
        self._timing_sync()
        elapsed = time.perf_counter() - started
        print(
            f"[timing][sample={self._timing_sample_id}]"
            f"[chunk={self._timing_chunk}][stage={stage}] {elapsed:.3f}s",
            file=sys.stderr,
            flush=True,
        )

    def reset(self) -> None:
        cfg = self.config
        self.manager = ComplementaryTokenMemory(
            qim_capacity=cfg.qim_capacity,
            im_capacity=cfg.im_capacity,
            active_buffer_capacity=cfg.active_buffer_capacity,
            im_episode_budget=cfg.im_episode_budget,
        )
        self.history: List[Dict[str, str]] = []
        self.dsh_state = None
        self.temporal_offset = 0
        self.turn_id = 0
        self.total_seen_frames = 0
        self._query_carrier_frames = None
        self._query_carrier_temporal_offset = None
        self._query_carrier_dsh = None
        self._query_carrier_source_turn = None

    def load_video(self, video_path: str, *, fps: Optional[float] = None, end_time: Optional[float] = None):
        cfg = self.config
        sampling = {
            "fps": cfg.fps if fps is None else fps,
            "min_frames": cfg.min_frames,
            "max_frames": cfg.max_frames,
            "min_pixels": cfg.min_pixels,
            "max_pixels": cfg.max_pixels,
            "end_time": end_time,
        }
        decoder = lambda: load_video_frames(video_path, **sampling)
        if self.frame_cache is None:
            return decoder()
        frames, _ = self.frame_cache.load_or_compute(video_path, sampling, decoder)
        return frames

    def _visual_backbone_features(self, frames, inputs):
        cached = None
        if self.feature_cache is not None:
            cached, _ = self.feature_cache.load(frames)
        if cached is not None:
            cached_grid = cached["video_grid_thw"]
            current_grid = inputs.video_grid_thw.detach().cpu()
            if not torch.equal(cached_grid, current_grid):
                raise ValueError("visual cache grid does not match current processor output")
            values = cached["qwen_values"].to(device=self.device, dtype=self.model_dtype)
            aligned = cached["aligned_semantic_tokens"]
            aligned = aligned.to(self.semantic_device)
            return values, aligned, True

        values, visual_type = self.qwen.embedding_visual_info(**inputs)
        if visual_type != "video":
            raise ValueError("Passive benchmark inference expected backbone video embeddings.")
        values = values.to(device=self.device, dtype=self.model_dtype)
        aligned, _ = encode_aligned_semantic_tokens(
            video_frames=list(frames),
            video_grid_thw=inputs.video_grid_thw.detach().cpu(),
            semantic_encoder=self.semantic_encoder,
            semantic_preprocess=self.semantic_preprocess,
            semantic_device=self.semantic_device,
        )
        if self.feature_cache is not None:
            self.feature_cache.store(frames, inputs.video_grid_thw, values, aligned)
        return values, aligned, False

    @torch.inference_mode()
    def precompute_chunk(self, frames: Sequence) -> bool:
        """Populate universal visual caches; return True when already cached."""
        messages = _build_messages([], list(frames), "Describe the visible video.")
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[prompt], images=None, videos=[list(frames)], padding=True, return_tensors="pt"
        ).to(self.device)
        _, _, hit = self._visual_backbone_features(frames, inputs)
        return hit

    def _encode_chunk(self, frames: Sequence, query: str):
        if self._stage_timing:
            self._timing_chunk += 1
        started = self._timing_start()
        messages = _build_messages(self.history, list(frames), query)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=None, videos=[list(frames)], padding=True, return_tensors="pt"
        ).to(self.device)
        self._timing_print("processor_and_transfer", started)

        started = self._timing_start()
        visual_values, aligned_semantic_tokens, feature_cache_hit = self._visual_backbone_features(frames, inputs)
        self._timing_print("qwen_vision", started)

        started = self._timing_start()
        features = build_visual_evidence(
            aligned_semantic_tokens=aligned_semantic_tokens,
            video_grid_thw=inputs.video_grid_thw.detach().cpu(),
            query=query,
            semantic_encoder=self.semantic_encoder,
            semantic_processor=self.semantic_tokenizer,
            semantic_device=self.semantic_device,
            temporal_alpha=self.config.temporal_alpha,
            novelty_weight=self.config.novelty_weight,
            fps=self.config.fps,
            global_temporal_offset=self.temporal_offset,
            initial_dsh=self.dsh_state,
            output_device=self.semantic_device,
        )
        if self._stage_timing:
            print(
                f"[timing][sample={self._timing_sample_id}][chunk={self._timing_chunk}]"
                f"[visual_feature_cache={'hit' if feature_cache_hit else 'miss'}]",
                file=sys.stderr,
                flush=True,
            )
        self._timing_print("siglip2_and_features", started)
        self.dsh_state = features.temporal_state

        started = self._timing_start()
        full_embeds = inject_visual_embeddings(self.model, inputs.input_ids, visual_values)
        active_budget = self.config.active_budget or max(
            1, int(round(visual_values.shape[0] * self.config.active_keep_rate))
        )
        router_scores, active_mask, selector = route_current_tokens(
            features=features,
            current_visual_values=visual_values,
            input_ids=inputs.input_ids,
            full_inputs_embeds=full_embeds,
            video_token_id=self.qwen.config.video_token_id,
            active_budget=active_budget,
            router=self.router,
        )
        self._timing_print("router", started)
        return inputs, visual_values, features, router_scores, active_mask, selector

    @torch.inference_mode()
    def ingest(self, frames: Sequence, query: str) -> Dict:
        """Consume a non-query chunk and update memory without generating text."""
        inputs, values, features, scores, active, selector = self._encode_chunk(frames, query)
        started = self._timing_start()
        self.manager.observe(
            semantic_keys=features.semantic_keys,
            qwen_values=values.detach(),
            novelty=features.novelty,
            router_scores=scores,
            active_mask=active,
            token_metadata=features.metadata,
            turn_id=self.turn_id,
        )
        self.manager.consolidate_turn(features.query_embedding, self.turn_id)
        self._timing_print("memory_update", started)
        self.temporal_offset += int(inputs.video_grid_thw[0, 0])
        self.total_seen_frames += len(frames)
        self.turn_id += 1
        return {"selector": selector, "selected_current_tokens": int(active.sum()), **self.manager.stats()}

    @torch.inference_mode()
    def answer(self, frames: Sequence, query: str) -> Dict:
        """Consume the current chunk, retrieve memory, and answer one passive query."""
        inputs, values, features, scores, active, selector = self._encode_chunk(frames, query)
        started = self._timing_start()
        recalled = self.manager.retrieve(
            query_embedding=features.query_embedding,
            qim_budget=self.config.qim_read_budget,
            im_budget=self.config.im_read_budget,
        )
        self._timing_print("memory_retrieve", started)

        started = self._timing_start()
        self.manager.observe(
            semantic_keys=features.semantic_keys,
            qwen_values=values.detach(),
            novelty=features.novelty,
            router_scores=scores,
            active_mask=active,
            token_metadata=features.metadata,
            turn_id=self.turn_id,
        )
        self._timing_print("memory_update", started)

        started = self._timing_start()
        packed = pack_visual_context(
            model=self.qwen,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            current_visual_values=values,
            current_keep_mask=active,
            current_metadata=features.metadata,
            recalled_slots=recalled.slots,
        )
        self.qwen.set_memory_prefill_positions(packed.position_ids, packed.sequence_length)
        self._timing_print("memory_pack", started)

        started = self._timing_start()
        generated = self.model.generate(
            input_ids=packed.input_ids,
            inputs_embeds=packed.inputs_embeds,
            attention_mask=packed.attention_mask,
            position_ids=packed.position_ids,
            use_cache=True,
            do_sample=False,
            max_new_tokens=self.config.max_new_tokens,
        )
        self._timing_print("generation", started)
        generated_only = (
            generated[:, packed.sequence_length :]
            if generated.shape[1] > packed.sequence_length
            else generated
        )
        started = self._timing_start()
        answer = self.processor.batch_decode(
            generated_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        self.manager.consolidate_turn(features.query_embedding, self.turn_id)
        self._timing_print("text_decode_and_consolidate", started)
        self.history.append({"query": query, "answer": answer})
        self.temporal_offset += int(inputs.video_grid_thw[0, 0])
        self.total_seen_frames += len(frames)
        self.turn_id += 1
        return {
            "answer": answer,
            "selector": selector,
            "selected_current_tokens": int(active.sum()),
            "recalled_qim_tokens": len(recalled.qim_slots),
            "recalled_im_tokens": len(recalled.im_slots),
            "packed_memory_tokens": packed.num_memory_tokens,
            "packed_current_tokens": packed.num_current_tokens,
            "seen_frames": self.total_seen_frames,
            **self.manager.stats(),
        }

    @torch.inference_mode()
    def answer_without_ingest(self, query: str) -> Dict:
        """Answer at the current timestamp without rewriting memory."""
        if self._query_carrier_frames is None:
            raise RuntimeError("No visible video state is available for a query-only turn.")

        current_temporal_offset = self.temporal_offset
        current_dsh = self.dsh_state
        try:
            self.temporal_offset = int(self._query_carrier_temporal_offset)
            self.dsh_state = self._query_carrier_dsh
            inputs, values, features, scores, active, selector = self._encode_chunk(
                self._query_carrier_frames, query
            )
        finally:
            # Keep the visual timeline unchanged.
            self.temporal_offset = current_temporal_offset
            self.dsh_state = current_dsh

        started = self._timing_start()
        recalled = self.manager.retrieve(
            query_embedding=features.query_embedding,
            qim_budget=self.config.qim_read_budget,
            im_budget=self.config.im_read_budget,
        )
        # Avoid packing the carrier segment twice.
        source_turn = self._query_carrier_source_turn
        recalled_slots = [
            slot for slot in recalled.slots if source_turn is None or slot.source_turn != source_turn
        ]
        self._timing_print("memory_retrieve", started)

        started = self._timing_start()
        packed = pack_visual_context(
            model=self.qwen,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            current_visual_values=values,
            current_keep_mask=active,
            current_metadata=features.metadata,
            recalled_slots=recalled_slots,
        )
        self.qwen.set_memory_prefill_positions(packed.position_ids, packed.sequence_length)
        self._timing_print("memory_pack", started)

        started = self._timing_start()
        generated = self.model.generate(
            input_ids=packed.input_ids,
            inputs_embeds=packed.inputs_embeds,
            attention_mask=packed.attention_mask,
            position_ids=packed.position_ids,
            use_cache=True,
            do_sample=False,
            max_new_tokens=self.config.max_new_tokens,
        )
        self._timing_print("generation", started)
        generated_only = (
            generated[:, packed.sequence_length :]
            if generated.shape[1] > packed.sequence_length
            else generated
        )
        answer = self.processor.batch_decode(
            generated_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        self.history.append({"query": query, "answer": answer})
        self.turn_id += 1
        recalled_qim = sum(slot.path == "qim" for slot in recalled_slots)
        recalled_im = sum(slot.path == "im" for slot in recalled_slots)
        return {
            "answer": answer,
            "selector": selector,
            "selected_current_tokens": int(active.sum()),
            "recalled_qim_tokens": recalled_qim,
            "recalled_im_tokens": recalled_im,
            "packed_memory_tokens": packed.num_memory_tokens,
            "packed_current_tokens": packed.num_current_tokens,
            "seen_frames": self.total_seen_frames,
            "query_only_turn": True,
            **self.manager.stats(),
        }

    def answer_video(self, video_path: str, query: str) -> Dict:
        """Passive single-question evaluation with chunked long-video memory."""
        self.reset()
        frames = self.load_video(video_path)
        return self.answer_interval(frames, query)

    def answer_interval(self, frames: Sequence, query: str) -> Dict:
        """Process one newly visible interval and generate exactly one answer."""
        if not frames:
            raise ValueError("A passive query interval must contain at least one new frame.")
        frames = list(frames)
        if len(frames) < 2:
            frames.append(frames[-1].copy())
        chunks = [frames[index : index + self.chunk_frames] for index in range(0, len(frames), self.chunk_frames)]
        if len(chunks[-1]) % 2:
            chunks[-1].append(chunks[-1][-1].copy())
        for chunk in chunks[:-1]:
            self.ingest(chunk, query)
        self._query_carrier_frames = list(chunks[-1])
        self._query_carrier_temporal_offset = self.temporal_offset
        self._query_carrier_dsh = self.dsh_state
        self._query_carrier_source_turn = self.turn_id
        return self.answer(chunks[-1], query)
