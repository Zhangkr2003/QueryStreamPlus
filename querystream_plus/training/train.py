import argparse
import hashlib
import json
import math
import os
import random
import re
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import av
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from PIL import Image
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

from querystream_plus.modeling.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from querystream_plus.modeling.qwen2_5_vl.modeling_qwen2_5_vl import (
    QWEN2_5_VL_VISION_ATTENTION_CLASSES,
    flash_attn_varlen_func,
)
from querystream_plus.inference.vision import (
    _add_semantic_encoder_args,
    _load_semantic_encoder,
    _pool_siglip2_features_to_qwen_grid,
    _prepare_siglip2_batch,
    _semantic_encoder_spec,
    _siglip2_patch_features,
    _siglip2_text_feature,
    _video_input_to_pil_frames,
    round_by_factor,
    smart_resize,
)
from querystream_plus.modeling.router import (
    RouterConfig,
    build_router,
    build_budgeted_keep_mask,
    normalize_prior_scores,
    build_router_attention_bias,
)
from querystream_plus.training.data import AnswerEvent, load_answer_events
from querystream_plus.memory.manager import ComplementaryTokenMemory
from querystream_plus.memory.features import encode_visual_evidence
from querystream_plus.memory.packing import pack_visual_context


def parse_args():
    parser = argparse.ArgumentParser(description="Train the QueryStream++ router and adapter.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-jsonl", required=True, help="Timeline-format JSONL training file.")
    parser.add_argument(
        "--media-root",
        default=None,
        help="Optional root for relative video/frame paths; defaults to the JSONL directory.",
    )
    parser.add_argument("--output-dir", required=True)
    _add_semantic_encoder_args(parser)
    parser.add_argument("--semantic-device", default="cpu")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        choices=["sdpa", "eager"],
        help=(
            "Attention backend for training. eager is the robust default because PyTorch 2.6 "
            "SDPA backward can fail with LSE stride alignment after router bias/pruning."
        ),
    )
    parser.add_argument(
        "--vision-attn-implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help=(
            "Attention backend used only by the frozen vision tower. The language decoder "
            "can remain eager for router bias/pruning while the much longer pre-merge visual "
            "sequence uses memory-efficient FlashAttention2."
        ),
    )
    parser.add_argument(
        "--allow-fused-sdpa-backward",
        action="store_true",
        help=(
            "Allow CUDA Flash/memory-efficient/cuDNN SDPA kernels during training. "
            "They are disabled by default to avoid LSE strideH backward failures."
        ),
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=1016)
    parser.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument(
        "--max-visual-tokens",
        type=int,
        default=8192,
        help=(
            "Dynamic per-event visual budget within the 11264-token context. "
            "Frames are reduced only when this budget is exceeded."
        ),
    )
    parser.add_argument("--qdp-align-method", default="overlap", choices=["overlap"])
    parser.add_argument("--temporal-alpha", type=float, default=0.1)
    parser.add_argument("--novelty-weight", type=float, default=1.0)
    parser.add_argument("--keep-rate", type=float, default=0.5)
    parser.add_argument("--router-layer", type=int, default=0)
    parser.add_argument("--prune-after-layer", type=int, default=0)
    parser.add_argument("--router-dim", type=int, default=768)
    parser.add_argument("--router-heads", type=int, default=12)
    parser.add_argument("--router-ffn-ratio", type=float, default=4.0)
    parser.add_argument("--router-spatial-window", type=int, default=4)
    parser.add_argument("--router-temporal-window", type=int, default=8)
    parser.add_argument("--router-dropout", type=float, default=0.0)
    parser.add_argument(
        "--training-stage",
        choices=("router", "joint"),
        default="router",
        help="router freezes the backbone; joint trains the router and decoder LoRA adapter.",
    )
    parser.add_argument(
        "--router-init-checkpoint",
        default=None,
        help="Stage-one router checkpoint used to initialize joint fine-tuning.",
    )
    parser.add_argument(
        "--lora-adapter-init",
        default=None,
        help="Optional standard PEFT adapter directory to continue joint fine-tuning from.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="QueryStream training checkpoint that restores router, adapter, optimizer, and scheduler.",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="LLM module suffixes. Defaults match ms-swift's LLM-only all-linear policy.",
    )
    parser.add_argument("--lora-lr", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--prior-reg-start", type=float, default=0.05)
    parser.add_argument("--prior-reg-end", type=float, default=0.0)
    parser.add_argument("--router-residual-warmup", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--print-token-stats", action="store_true")
    parser.add_argument(
        "--skip-bad-samples",
        action="store_true",
        help=(
            "On a single-GPU run, log and skip samples that fail during media/frame "
            "preparation. CUDA, model-forward, and optimizer errors are never swallowed."
        ),
    )
    parser.add_argument(
        "--bad-sample-log",
        default=None,
        help="Optional JSONL path for skipped data errors; defaults under --output-dir.",
    )
    parser.add_argument(
        "--distributed-timeout-seconds",
        type=int,
        default=3600,
        help="DDP process-group timeout; long-video ranks can legitimately diverge in runtime.",
    )
    parser.add_argument(
        "--memory-aware",
        action="store_true",
        help=(
            "Enable deterministic QIM/IM context construction for samples whose "
            "training_mode is 'memory'. Joint stage only; memory values are detached."
        ),
    )
    parser.add_argument("--qim-capacity", type=int, default=1024)
    parser.add_argument("--im-capacity", type=int, default=2048)
    parser.add_argument("--active-buffer-capacity", type=int, default=2048)
    parser.add_argument("--im-episode-budget", type=int, default=128)
    parser.add_argument("--qim-read-budget", type=int, default=512)
    parser.add_argument("--im-read-budget", type=int, default=512)
    parser.add_argument(
        "--chunkwise-memory-training",
        action="store_true",
        help=(
            "For memory events, stream every newly visible interval through fixed-size "
            "chunks. Historical chunks update detached QIM/IM state; only the final "
            "chunk retrieves memory and computes the answer loss."
        ),
    )
    parser.add_argument(
        "--stream-chunk-frames",
        type=int,
        default=16,
        help="Even number of sampled frames processed by each streaming memory chunk.",
    )
    parser.add_argument(
        "--hybrid-memory-training",
        action="store_true",
        help=(
            "Use direct router pruning for a short first turn and chunk-wise memory for "
            "long first turns or every later turn. Memory samples still write the first "
            "short turn after its answer so later turns can retrieve it."
        ),
    )
    parser.add_argument(
        "--mixed-visual-budgets",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Deterministic per-event post-processor visual budgets. For example, "
            "`4096 6144` trains one checkpoint for both 4K and 6K inference."
        ),
    )
    parser.add_argument(
        "--mixed-qim-read-budgets",
        type=int,
        nargs="+",
        default=None,
        help="QIM retrieval budget paired with each --mixed-visual-budgets entry.",
    )
    parser.add_argument(
        "--mixed-im-read-budgets",
        type=int,
        nargs="+",
        default=None,
        help="IM retrieval budget paired with each --mixed-visual-budgets entry.",
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def configure_cuda_sdpa_for_training(allow_fused: bool) -> Dict[str, Optional[bool]]:
    """Force the stable math kernel unless fused SDPA backward is explicitly enabled."""
    if torch.cuda.is_available() and not allow_fused:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)

    def read_flag(name: str) -> Optional[bool]:
        function = getattr(torch.backends.cuda, name, None)
        return None if function is None else bool(function())

    return {
        "flash": read_flag("flash_sdp_enabled"),
        "memory_efficient": read_flag("mem_efficient_sdp_enabled"),
        "math": read_flag("math_sdp_enabled"),
        "cudnn": read_flag("cudnn_sdp_enabled"),
    }


def replace_frozen_vision_attention(model, implementation: str) -> None:
    """Set the frozen vision tower's attention backend."""
    if implementation == "flash_attention_2" and flash_attn_varlen_func is None:
        raise RuntimeError(
            "--vision-attn-implementation flash_attention_2 requires FlashAttention2. "
            "Install it with `pip install flash-attn==2.7.4.post1 --no-build-isolation`, "
            "or use --vision-attn-implementation sdpa for a lower-memory debug run."
        )
    qwen = base_qwen_model(model)
    vision = qwen.visual
    attention_class = QWEN2_5_VL_VISION_ATTENTION_CLASSES[implementation]
    config = vision.config
    for block in vision.blocks:
        old_attention = block.attn
        if isinstance(old_attention, attention_class):
            continue
        new_attention = attention_class(config.hidden_size, num_heads=config.num_heads)
        new_attention.load_state_dict(old_attention.state_dict(), strict=True)
        reference = old_attention.qkv.weight
        block.attn = new_attention.to(device=reference.device, dtype=reference.dtype)
    config._attn_implementation = implementation


def _import_peft():
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "Joint training requires peft==0.17.1 with accelerate>=0.32.0. "
            "Install the project requirements or run "
            "`pip install peft==0.17.1 accelerate==0.34.2`."
        ) from exc
    return LoraConfig, PeftModel, TaskType, get_peft_model


def base_qwen_model(model):
    """Return the QueryStream Qwen model under an optional PEFT wrapper."""
    if isinstance(model, DistributedDataParallel):
        model = model.module
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def unwrap_distributed(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def initialize_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, 0
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed QueryStream training currently requires CUDA/NCCL.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=args.distributed_timeout_seconds),
    )
    args.device = f"cuda:{local_rank}"
    if str(args.semantic_device).startswith("cuda"):
        args.semantic_device = f"cuda:{local_rank}"
    return dist.get_rank(), dist.get_world_size(), local_rank


def language_lora_target_regex(target_modules: List[str]) -> str:
    """Restrict suffix-style targets to Qwen language-decoder layers, excluding ViT/merger."""
    if not target_modules:
        raise ValueError("--lora-target-modules must contain at least one module suffix.")
    module_pattern = "|".join(re.escape(name) for name in target_modules)
    return rf"^model\.layers\.\d+\.(?:self_attn|mlp)\.(?:{module_pattern})$"


def validate_training_args(args):
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1).")
    if args.training_stage == "router" and args.lora_adapter_init:
        raise ValueError("--lora-adapter-init is only valid for --training-stage joint.")
    if args.training_stage == "joint" and not (args.router_init_checkpoint or args.resume_checkpoint):
        raise ValueError(
            "Joint training must start from a stage-one router via --router-init-checkpoint, "
            "or restore a joint run via --resume-checkpoint."
        )
    if args.resume_checkpoint and args.lora_adapter_init:
        raise ValueError("Use either --resume-checkpoint or --lora-adapter-init, not both.")
    if args.resume_checkpoint and args.router_init_checkpoint:
        raise ValueError("Use either --resume-checkpoint or --router-init-checkpoint, not both.")
    if args.memory_aware and args.training_stage != "joint":
        raise ValueError("--memory-aware is only supported for --training-stage joint.")
    if args.chunkwise_memory_training and not args.memory_aware:
        raise ValueError("--chunkwise-memory-training requires --memory-aware.")
    if args.hybrid_memory_training and not args.memory_aware:
        raise ValueError("--hybrid-memory-training requires --memory-aware.")
    if args.hybrid_memory_training and not args.chunkwise_memory_training:
        raise ValueError("--hybrid-memory-training requires --chunkwise-memory-training.")
    if args.stream_chunk_frames < 2 or args.stream_chunk_frames % 2 != 0:
        raise ValueError("--stream-chunk-frames must be an even integer of at least two.")
    for name in ("qim_capacity", "im_capacity", "qim_read_budget", "im_read_budget"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if args.max_visual_tokens <= 0:
        raise ValueError("--max-visual-tokens must be positive.")
    if args.distributed_timeout_seconds <= 0:
        raise ValueError("--distributed-timeout-seconds must be positive.")
    mixed_groups = (
        args.mixed_visual_budgets,
        args.mixed_qim_read_budgets,
        args.mixed_im_read_budgets,
    )
    if any(group is not None for group in mixed_groups):
        if not all(group is not None for group in mixed_groups):
            raise ValueError(
                "Mixed-budget training requires --mixed-visual-budgets, "
                "--mixed-qim-read-budgets, and --mixed-im-read-budgets together."
            )
        lengths = {len(group) for group in mixed_groups}
        if lengths != {len(args.mixed_visual_budgets)} or not args.mixed_visual_budgets:
            raise ValueError("All mixed-budget lists must be non-empty and have equal length.")
        for visual, qim, im in zip(*mixed_groups):
            if visual <= 0 or qim < 0 or im < 0:
                raise ValueError("Mixed visual budgets must be positive; read budgets non-negative.")
            if qim + im > visual:
                raise ValueError(
                    f"Mixed QIM+IM reads ({qim + im}) exceed visual budget {visual}."
                )
    tokens_per_pair = max(1, int(math.ceil(args.max_pixels / float(28 * 28))))
    chunk_tokens = (args.stream_chunk_frames // 2) * tokens_per_pair
    if args.chunkwise_memory_training and chunk_tokens > args.max_visual_tokens:
        raise ValueError(
            f"A {args.stream_chunk_frames}-frame chunk needs about {chunk_tokens} visual "
            f"tokens, exceeding --max-visual-tokens={args.max_visual_tokens}."
        )


def build_messages(event: AnswerEvent, video_frames: List) -> List[Dict]:
    messages = []
    current_user_index = next(
        (
            index
            for index in range(len(event.dialogue_context) - 1, -1, -1)
            if event.dialogue_context[index].get("role") == "user"
        ),
        None,
    )
    for index, msg in enumerate(event.dialogue_context):
        role = msg["role"]
        text = msg["text"]
        if index == current_user_index:
            messages.append(
                {
                    "role": role,
                    "content": [
                        {"type": "video", "video": video_frames},
                        {"type": "text", "text": text},
                    ],
                }
            )
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})
    if current_user_index is None:
        messages.insert(0, {"role": "user", "content": [{"type": "video", "video": video_frames}]})
    return messages


def target_text(event: AnswerEvent) -> str:
    target = event.target
    if target.get("mode") == "multi_reference":
        return random.choice(target["answers"])
    return target["text"]


def _training_target_frame_count(
    duration: float,
    args,
    available_frames: Optional[int] = None,
    visual_token_budget: Optional[int] = None,
) -> int:
    """Compute the adaptive frame budget before materializing any pixels."""
    tokens_per_pair = max(1, int(math.ceil(args.max_pixels / float(28 * 28))))
    token_budget = args.max_visual_tokens if visual_token_budget is None else visual_token_budget
    budget_frames = max(2, 2 * (token_budget // tokens_per_pair))
    upper_bound = min(args.max_frames, budget_frames)
    if available_frames is not None:
        upper_bound = min(upper_bound, max(1, int(available_frames)))

    target_frames = min(max(duration * args.fps, args.min_frames), upper_bound)
    target_frames = max(1, int(target_frames))
    if target_frames >= 2:
        target_frames = max(2, round_by_factor(target_frames, 2))
        target_frames = min(target_frames, upper_bound)
        if target_frames % 2 == 1 and target_frames > 2:
            target_frames -= 1
    return max(1, target_frames)


def _resize_training_frame(frame: Image.Image, args) -> Image.Image:
    frame = frame.convert("RGB")
    width, height = frame.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    return frame.resize((resized_width, resized_height), Image.Resampling.BICUBIC)


def _decode_uniform_video_frames(
    video_path: str,
    start: float,
    end: float,
    target_frames: int,
    args,
) -> List[Image.Image]:
    """Uniformly sample a causal interval while retaining bounded host memory."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    source_fps = float(stream.average_rate) if stream.average_rate else 1.0
    if start > 0.0 and stream.time_base is not None:
        try:
            # Timestamp checks still enforce the causal interval.
            container.seek(
                max(0, int(start / float(stream.time_base))),
                stream=stream,
                any_frame=False,
                backward=True,
            )
        except Exception:
            # Fall back to sequential decoding.
            container.close()
            container = av.open(video_path)
            stream = container.streams.video[0]
            source_fps = float(stream.average_rate) if stream.average_rate else 1.0

    requested_times = (
        [end]
        if target_frames <= 1 or end <= start
        else torch.linspace(start, end, target_frames).tolist()
    )
    sampled: List[Image.Image] = []
    target_index = 0
    last_visible_frame = None
    for frame_idx, frame in enumerate(container.decode(stream)):
        timestamp = (
            float(frame.pts * frame.time_base)
            if frame.pts is not None and frame.time_base is not None
            else frame_idx / max(source_fps, 1e-6)
        )
        if timestamp < start or (start > 0.0 and timestamp <= start):
            continue
        if timestamp > end:
            break
        last_visible_frame = frame
        if target_index < len(requested_times) and timestamp >= requested_times[target_index]:
            selected = _resize_training_frame(frame.to_image(), args)
            while target_index < len(requested_times) and timestamp >= requested_times[target_index]:
                sampled.append(selected if not sampled else selected.copy())
                target_index += 1

    trailing_image = (
        last_visible_frame.to_image()
        if target_index < len(requested_times) and last_visible_frame is not None
        else None
    )
    container.close()
    if target_index < len(requested_times) and trailing_image is not None:
        selected = _resize_training_frame(trailing_image, args)
        while target_index < len(requested_times):
            sampled.append(selected if not sampled else selected.copy())
            target_index += 1
    if not sampled:
        raise ValueError(f"No frames decoded for visible span [{start}, {end}] in {video_path}")
    while len(sampled) < max(2, args.min_frames):
        sampled.append(sampled[-1].copy())
    if len(sampled) % 2 == 1:
        sampled.append(sampled[-1].copy())
    return sampled


def sample_visible_frames(
    event: AnswerEvent,
    args,
    force_causal_prefix: bool = False,
    visual_token_budget: Optional[int] = None,
) -> List:
    start = 0.0 if force_causal_prefix else max(float(event.visible_video_start), 0.0)
    end = max(float(event.visible_video_end), start)
    source_fps = max(float(args.fps), 1e-6)
    if event.frames:
        timestamps = event.frame_timestamps
        if timestamps is None:
            timestamps = [index / source_fps for index in range(len(event.frames))]
        selected_paths = [
            path
            for path, timestamp in zip(event.frames, timestamps)
            if (float(timestamp) >= start if start == 0.0 else float(timestamp) > start)
            and float(timestamp) <= end
        ]
        if not selected_paths and event.frames:
            causal_candidates = [
                index for index, timestamp in enumerate(timestamps) if float(timestamp) <= end
            ]
            if not causal_candidates:
                raise ValueError(
                    f"No causally visible frame is available at t={end} for event {event.event_id}."
                )
            nearest = max(causal_candidates, key=lambda index: float(timestamps[index]))
            selected_paths = [event.frames[nearest]]
        duration = max(end - start, len(selected_paths) / source_fps)
        target_frames = _training_target_frame_count(
            duration, args, len(selected_paths), visual_token_budget
        )
        indices = torch.linspace(0, len(selected_paths) - 1, target_frames).round().long().tolist()
        sampled = [_resize_training_frame(Image.open(selected_paths[index]), args) for index in indices]
    else:
        duration = max(end - start, 1.0 / source_fps)
        target_frames = _training_target_frame_count(
            duration, args, visual_token_budget=visual_token_budget
        )
        return _decode_uniform_video_frames(event.video, start, end, target_frames, args)

    while len(sampled) < args.min_frames:
        sampled.append(sampled[-1].copy())
    if len(sampled) % 2 == 1:
        sampled.append(sampled[-1].copy())

    return sampled


def _stream_target_frame_count(duration: float, args) -> int:
    """Number of 1-FPS-style frames retained across a complete event interval."""
    requested = max(args.min_frames, int(math.ceil(max(duration, 0.0) * args.fps)))
    return max(2, min(requested, args.max_frames))


def choose_event_budgets(event: AnswerEvent, epoch: int, args) -> Tuple[int, int, int]:
    """Choose a reproducible budget independent of rank and Python hash randomization."""
    if args.mixed_visual_budgets is None:
        return args.max_visual_tokens, args.qim_read_budget, args.im_read_budget
    identity = f"{args.seed}:{epoch}:{event.sample_id}:{event.event_id}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % len(
        args.mixed_visual_budgets
    )
    return (
        args.mixed_visual_budgets[index],
        args.mixed_qim_read_budgets[index],
        args.mixed_im_read_budgets[index],
    )


def estimate_interval_visual_tokens(event: AnswerEvent, args) -> int:
    """Estimate merged Qwen tokens for the causally new interval at streaming FPS."""
    duration = max(float(event.visible_video_end) - float(event.visible_video_start), 0.0)
    frame_count = _stream_target_frame_count(duration, args)
    if event.frames:
        timestamps = event.frame_timestamps
        if timestamps is None:
            timestamps = [index / max(float(args.fps), 1e-6) for index in range(len(event.frames))]
        start = max(float(event.visible_video_start), 0.0)
        end = max(float(event.visible_video_end), start)
        available = sum(
            (float(timestamp) >= start if start == 0.0 else float(timestamp) > start)
            and float(timestamp) <= end
            for timestamp in timestamps
        )
        if available:
            frame_count = min(frame_count, available)
    tokens_per_pair = max(1, int(math.ceil(args.max_pixels / float(28 * 28))))
    return int(math.ceil(frame_count / 2.0) * tokens_per_pair)


def hybrid_event_policy(
    event: AnswerEvent,
    epoch: int,
    args,
) -> Tuple[str, int, int, int, int]:
    """Return policy, total visual budget, QIM/IM reads, and raw token estimate."""
    visual_budget, qim_budget, im_budget = choose_event_budgets(event, epoch, args)
    estimated_tokens = estimate_interval_visual_tokens(event, args)
    if not args.hybrid_memory_training:
        policy = "chunk_memory" if event.training_mode == "memory" else "direct_prune"
    elif event.turn_index > 0 or estimated_tokens > visual_budget:
        policy = "chunk_memory"
    else:
        policy = "direct_prune"
    return policy, visual_budget, qim_budget, im_budget, estimated_tokens


def _chunk_resized_frames(
    frames: Sequence[Image.Image],
    chunk_frames: int,
) -> Iterator[List[Image.Image]]:
    pending: List[Image.Image] = []
    for frame in frames:
        pending.append(frame)
        if len(pending) == chunk_frames:
            yield pending
            pending = []
    if pending:
        if len(pending) % 2 == 1:
            pending.append(pending[-1].copy())
        yield pending


def _iter_video_frame_chunks(
    video_path: str,
    start: float,
    end: float,
    args,
) -> Iterator[List[Image.Image]]:
    """Yield uniformly sampled, resized chunks without retaining the whole prefix."""
    duration = max(end - start, 1.0 / max(args.fps, 1e-6))
    target_frames = _stream_target_frame_count(duration, args)
    requested_times = torch.linspace(start, end, target_frames).tolist()

    container = av.open(video_path)
    stream = container.streams.video[0]
    source_fps = float(stream.average_rate) if stream.average_rate else 1.0
    if start > 0.0 and stream.time_base is not None:
        try:
            container.seek(
                max(0, int(start / float(stream.time_base))),
                stream=stream,
                any_frame=False,
                backward=True,
            )
        except Exception:
            container.close()
            container = av.open(video_path)
            stream = container.streams.video[0]
            source_fps = float(stream.average_rate) if stream.average_rate else 1.0

    pending: List[Image.Image] = []
    target_index = 0
    last_visible_frame = None
    try:
        for frame_index, frame in enumerate(container.decode(stream)):
            timestamp = (
                float(frame.pts * frame.time_base)
                if frame.pts is not None and frame.time_base is not None
                else frame_index / max(source_fps, 1e-6)
            )
            if timestamp < start or (start > 0.0 and timestamp <= start):
                continue
            if timestamp > end:
                break
            last_visible_frame = frame
            if target_index >= len(requested_times) or timestamp < requested_times[target_index]:
                continue
            selected = _resize_training_frame(frame.to_image(), args)
            while target_index < len(requested_times) and timestamp >= requested_times[target_index]:
                pending.append(selected if not pending else selected.copy())
                target_index += 1
                if len(pending) == args.stream_chunk_frames:
                    yield pending
                    pending = []

        if target_index < len(requested_times) and last_visible_frame is not None:
            selected = _resize_training_frame(last_visible_frame.to_image(), args)
            while target_index < len(requested_times):
                pending.append(selected if not pending else selected.copy())
                target_index += 1
                if len(pending) == args.stream_chunk_frames:
                    yield pending
                    pending = []
    finally:
        container.close()

    if not pending and target_index == 0:
        # Reuse the last causal frame when annotation times exceed the stream.
        fallback_end = end
        probe = av.open(video_path)
        try:
            probe_stream = probe.streams.video[0]
            if probe_stream.duration is not None and probe_stream.time_base is not None:
                encoded_end = float(probe_stream.duration * probe_stream.time_base)
                if encoded_end > 0:
                    fallback_end = min(fallback_end, encoded_end)
            elif probe.duration is not None:
                encoded_end = float(probe.duration) / float(av.time_base)
                if encoded_end > 0:
                    fallback_end = min(fallback_end, encoded_end)
        finally:
            probe.close()
        fallback_start = max(
            0.0,
            fallback_end - max(1.0, 1.0 / max(float(args.fps), 1e-6)),
        )
        fallback = _decode_uniform_video_frames(
            video_path,
            fallback_start,
            fallback_end,
            max(2, args.min_frames),
            args,
        )
        print(
            json.dumps(
                {
                    "warning": "video_tail_frame_fallback",
                    "video": video_path,
                    "requested_span": [start, end],
                    "fallback_span": [fallback_start, fallback_end],
                    "frames": len(fallback),
                }
            ),
            flush=True,
        )
        yield fallback
        return
    if pending:
        if len(pending) % 2 == 1:
            pending.append(pending[-1].copy())
        yield pending


def iter_visible_frame_chunks(event: AnswerEvent, args) -> Iterator[List[Image.Image]]:
    """Stream one causal answer interval using the same chunk contract as inference."""
    start = max(float(event.visible_video_start), 0.0)
    end = max(float(event.visible_video_end), start)
    if not event.frames:
        yield from _iter_video_frame_chunks(event.video, start, end, args)
        return

    source_fps = max(float(args.fps), 1e-6)
    timestamps = event.frame_timestamps
    if timestamps is None:
        timestamps = [index / source_fps for index in range(len(event.frames))]
    candidates = [
        index
        for index, timestamp in enumerate(timestamps)
        if (float(timestamp) >= start if start == 0.0 else float(timestamp) > start)
        and float(timestamp) <= end
    ]
    if not candidates:
        causal = [index for index, timestamp in enumerate(timestamps) if float(timestamp) <= end]
        if not causal:
            raise ValueError(f"No causally visible frame is available at t={end} for {event.event_id}.")
        candidates = [max(causal, key=lambda index: float(timestamps[index]))]

    duration = max(end - start, len(candidates) / source_fps)
    target_frames = min(_stream_target_frame_count(duration, args), len(candidates))
    selected_indices = (
        torch.linspace(0, len(candidates) - 1, target_frames).round().long().tolist()
    )
    resized = (
        _resize_training_frame(Image.open(event.frames[candidates[index]]), args)
        for index in selected_indices
    )
    yield from _chunk_resized_frames(resized, args.stream_chunk_frames)


def build_inputs(processor, event: AnswerEvent, video_frames: List, answer: str):
    messages = build_messages(event, video_frames)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    eos = processor.tokenizer.eos_token or ""
    full_text = prompt_text + answer + eos

    inputs = processor(text=[full_text], images=None, videos=[video_frames], padding=True, return_tensors="pt")
    prompt_inputs = processor(text=[prompt_text], images=None, videos=[video_frames], padding=True, return_tensors="pt")
    labels = inputs.input_ids.clone()
    prompt_len = prompt_inputs.input_ids.shape[1]
    labels[:, :prompt_len] = -100
    labels[inputs.attention_mask == 0] = -100
    return inputs, labels, prompt_len


def current_query_text(event: AnswerEvent) -> str:
    for message in reversed(event.dialogue_context):
        if message.get("role") == "user" and str(message.get("text", "")).strip():
            return str(message["text"])
    return "\n".join(str(message.get("text", "")) for message in event.dialogue_context)


def ordered_epoch_events(events: List[AnswerEvent], seed: int) -> List[AnswerEvent]:
    """Shuffle sessions while preserving causal turn order inside memory sessions."""
    units: List[List[AnswerEvent]] = []
    memory_by_sample: Dict[str, List[AnswerEvent]] = {}
    for event in events:
        if event.training_mode == "memory":
            memory_by_sample.setdefault(event.sample_id, []).append(event)
        else:
            units.append([event])
    for session_events in memory_by_sample.values():
        session_events.sort(key=lambda item: (item.turn_index, item.visible_video_end, item.event_id))
        units.append(session_events)
    random.Random(seed).shuffle(units)
    return [event for unit in units for event in unit]


def distributed_epoch_events(
    events: List[AnswerEvent], seed: int, rank: int, world_size: int
) -> List[AnswerEvent]:
    """Shard whole causal sessions and pad ranks to the same DDP step count."""
    if world_size == 1:
        return ordered_epoch_events(events, seed)
    units: List[List[AnswerEvent]] = []
    memory_by_sample: Dict[str, List[AnswerEvent]] = {}
    for event in events:
        if event.training_mode == "memory":
            memory_by_sample.setdefault(event.sample_id, []).append(event)
        else:
            units.append([event])
    for session in memory_by_sample.values():
        session.sort(key=lambda item: (item.turn_index, item.visible_video_end, item.event_id))
        units.append(session)
    random.Random(seed).shuffle(units)
    local = [event for unit in units[rank::world_size] for event in unit]
    if not local:
        raise ValueError("The training set has fewer causal units than distributed ranks.")
    local_count = torch.tensor([len(local)], device=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}"))
    dist.all_reduce(local_count, op=dist.ReduceOp.MAX)
    target = int(local_count.item())
    # Padding uses a standalone causal event so it cannot corrupt a memory session.
    fallback = next((event for event in events if event.training_mode != "memory"), None)
    if fallback is None and len(local) < target:
        raise ValueError("DDP padding requires at least one non-memory training event.")
    local.extend([fallback] * (target - len(local)))
    return local


def packed_training_labels(
    original_input_ids: torch.LongTensor,
    original_labels: torch.LongTensor,
    packed_input_ids: torch.LongTensor,
    video_token_id: int,
) -> torch.LongTensor:
    """Remap answer labels after replacing the current video span by memory+selected tokens."""
    if original_input_ids.shape[0] != 1 or packed_input_ids.shape[0] != 1:
        raise ValueError("Memory-aware training currently supports batch size 1.")
    positions = (original_input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise ValueError("The original training input contains no video tokens.")
    prefix_end = int(positions[0])
    suffix_start = int(positions[-1]) + 1
    visual_labels = original_labels.new_full(
        (1, int((packed_input_ids[0] == video_token_id).sum().item())), -100
    )
    remapped = torch.cat(
        [original_labels[:, :prefix_end], visual_labels, original_labels[:, suffix_start:]], dim=1
    )
    if remapped.shape != packed_input_ids.shape:
        raise AssertionError("Packed labels do not align with packed input ids.")
    return remapped


@torch.no_grad()
def build_continuous_prior(
    video_frames: List,
    video_grid_thw: torch.Tensor,
    dialogue_context_text: str,
    semantic_encoder,
    semantic_preprocess,
    semantic_processor,
    semantic_device: torch.device,
    temporal_alpha: float,
    novelty_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid_t, grid_h, grid_w = [int(x) for x in video_grid_thw[0].tolist()]
    qwen_hw = (grid_h // 2, grid_w // 2)
    frames = _video_input_to_pil_frames(video_frames)
    semantic_images = _prepare_siglip2_batch(frames, semantic_preprocess, semantic_device)
    patch_features = _siglip2_patch_features(semantic_encoder, semantic_images)
    patch_features = F.normalize(patch_features, dim=-1)
    query_feature = _siglip2_text_feature(
        semantic_encoder, semantic_processor, dialogue_context_text, semantic_device
    )

    pooled = torch.stack([_pool_siglip2_features_to_qwen_grid(frame_feats, qwen_hw) for frame_feats in patch_features], dim=0)
    pooled = F.normalize(pooled, dim=-1)
    frame_indices = [min(i * 2 + 1, pooled.shape[0] - 1) for i in range(grid_t)]

    raw_scores = []
    rel_values = []
    novelty_values = []
    dsh = None
    for frame_idx in frame_indices:
        feats = pooled[frame_idx]
        relevance = feats @ query_feature
        if dsh is None:
            novelty = torch.ones_like(relevance)
            dsh = feats.clone()
        else:
            novelty = 1.0 - (feats * F.normalize(dsh, dim=-1)).sum(dim=-1)
            dsh = temporal_alpha * feats + (1.0 - temporal_alpha) * dsh
        raw = relevance + novelty_weight * novelty
        raw_scores.append(raw)
        rel_values.append(relevance)
        novelty_values.append(novelty)

    raw_scores = torch.cat(raw_scores, dim=0)
    relevance = torch.cat(rel_values, dim=0)
    novelty = torch.cat(novelty_values, dim=0)
    prior_score = normalize_prior_scores(raw_scores.unsqueeze(0)).squeeze(0)
    prior_features = torch.stack([prior_score, relevance, novelty], dim=-1)
    semantic_hidden = torch.cat([pooled[index] for index in frame_indices], dim=0)
    return prior_score.cpu(), prior_features.cpu(), semantic_hidden.cpu()


def make_inputs_embeds(model, inputs, visual_embedded, visual_embedded_type):
    model = base_qwen_model(model)
    input_ids = inputs.input_ids
    inputs_embeds = model.model.embed_tokens(input_ids)
    if visual_embedded_type != "video":
        raise ValueError("Router training currently expects video events.")
    mask = input_ids == model.config.video_token_id
    video_embeds = visual_embedded.to(inputs_embeds.device, inputs_embeds.dtype)
    if mask.sum().item() != video_embeds.shape[0]:
        video_embeds = video_embeds[: mask.sum().item()]
    inputs_embeds = inputs_embeds.masked_scatter(mask.unsqueeze(-1).expand_as(inputs_embeds), video_embeds)
    return inputs_embeds


def router_text_tokens(inputs_embeds, input_ids, prompt_len: int, video_token_id: int):
    prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    prompt_mask[:, :prompt_len] = True
    prompt_mask &= input_ids != video_token_id
    prompt_mask &= input_ids != 0
    # Keep prompt tokens for query-aware visual attention.
    selected = inputs_embeds[0, prompt_mask[0]].unsqueeze(0)
    if selected.shape[1] == 0:
        raise ValueError("No textual prompt token is available to condition the router.")
    return selected, torch.ones(selected.shape[:2], dtype=torch.bool, device=selected.device)


def _load_torch_checkpoint(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)


def _resume_adapter_path(resume_checkpoint: str, checkpoint: Dict) -> str:
    adapter = checkpoint.get("lora_adapter")
    if not adapter:
        raise ValueError("Joint resume checkpoint does not reference a saved LoRA adapter.")
    adapter_path = Path(adapter)
    if not adapter_path.is_absolute():
        adapter_path = Path(resume_checkpoint).resolve().parent / adapter_path
    if not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter referenced by resume checkpoint was not found: {adapter_path}")
    return str(adapter_path)


def _next_data_position(epoch: int, event_index: int, num_events: int) -> Tuple[int, int]:
    if event_index + 1 >= num_events:
        return epoch + 1, 0
    return epoch, event_index + 1


def save_training_checkpoint(
    router,
    model,
    optimizer,
    scheduler,
    args,
    global_step: int,
    optimizer_step: int,
    next_epoch: int,
    next_event_index: int,
    memory_manager=None,
    memory_dsh_state=None,
    memory_sample_id=None,
    memory_stream_turn_id: int = 0,
):
    router = unwrap_distributed(router)
    model = unwrap_distributed(model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_relative = None
    if args.training_stage == "joint":
        adapter_dir = out_dir / f"lora_step{global_step}"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        adapter_relative = adapter_dir.name
    path = out_dir / f"querystream_{args.training_stage}_step{global_step}.pt"
    torch.save(
        {
            "training_stage": args.training_stage,
            "step": global_step,
            "optimizer_step": optimizer_step,
            "next_epoch": next_epoch,
            "next_event_index": next_event_index,
            "router": router.state_dict(),
            "router_config": router.config.to_dict(),
            "lora_adapter": adapter_relative,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "memory_runtime": None
            if memory_manager is None
            else {
                "sample_id": memory_sample_id,
                "manager": memory_manager.state_dict(),
                "dsh_state": None if memory_dsh_state is None else memory_dsh_state.detach().cpu(),
                "stream_turn_id": int(memory_stream_turn_id),
            },
            "args": vars(args),
        },
        path,
    )
    return path


def main():
    args = parse_args()
    validate_training_args(args)
    rank, world_size, local_rank = initialize_distributed(args)
    if args.skip_bad_samples and world_size > 1:
        raise ValueError(
            "--skip-bad-samples supports one GPU only because DDP ranks must stay synchronized."
        )
    sdp_backends = configure_cuda_sdpa_for_training(args.allow_fused_sdpa_backward)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    device = torch.device(args.device)
    semantic_device = torch.device(args.semantic_device)
    dtype = resolve_dtype(args.torch_dtype)

    events = load_answer_events(args.train_jsonl)
    if not events:
        raise ValueError("No answer events were produced from the training JSONL.")
    data_root = Path(args.media_root).resolve() if args.media_root else Path(args.train_jsonl).resolve().parent
    for event in events:
        if not os.path.isabs(event.video):
            event.video = str((data_root / event.video).resolve())
        if event.frames:
            event.frames = [
                path if os.path.isabs(path) else str((data_root / path).resolve())
                for path in event.frames
            ]
    memory_event_count = sum(event.training_mode == "memory" for event in events)
    if memory_event_count and not args.memory_aware and rank == 0:
        print(
            json.dumps(
                {
                    "warning": "memory samples will use causal-prefix fallback because --memory-aware is disabled",
                    "memory_events": memory_event_count,
                }
            ),
            flush=True,
        )

    resume_state = _load_torch_checkpoint(args.resume_checkpoint) if args.resume_checkpoint else None
    if resume_state is not None and resume_state.get("training_stage", "router") != args.training_stage:
        raise ValueError("The resume checkpoint training stage does not match --training-stage.")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    replace_frozen_vision_attention(model, args.vision_attn_implementation)

    if args.training_stage == "joint":
        LoraConfig, PeftModel, TaskType, get_peft_model = _import_peft()
        adapter_init = args.lora_adapter_init
        if resume_state is not None:
            adapter_init = _resume_adapter_path(args.resume_checkpoint, resume_state)
        if adapter_init:
            model = PeftModel.from_pretrained(model, adapter_init, is_trainable=True)
        else:
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                target_modules=language_lora_target_regex(args.lora_target_modules),
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
        model.train()
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        base_qwen_model(model).visual.eval()
    else:
        # Training mode enables gradient checkpointing through the frozen decoder.
        model.train()
        base_qwen_model(model).visual.eval()
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

    processor = AutoProcessor.from_pretrained(args.model_path)
    semantic_model, semantic_checkpoint = _semantic_encoder_spec(args)
    semantic_encoder, semantic_preprocess, semantic_processor = _load_semantic_encoder(
        semantic_model, semantic_checkpoint, semantic_device
    )
    qwen = base_qwen_model(model)
    decoder_attention_class = type(qwen.model.layers[0].self_attn).__name__
    semantic_size = int(getattr(semantic_encoder.config.vision_config, "hidden_size", 1024))
    requested_router_config = RouterConfig(
        hidden_size=qwen.config.hidden_size,
        semantic_size=semantic_size,
        router_dim=args.router_dim,
        num_heads=args.router_heads,
        ffn_ratio=args.router_ffn_ratio,
        spatial_window=args.router_spatial_window,
        temporal_window=args.router_temporal_window,
        dropout=args.router_dropout,
    )
    router_init_state = _load_torch_checkpoint(args.router_init_checkpoint) if args.router_init_checkpoint else None
    saved_router_config = None
    if resume_state is not None:
        saved_router_config = resume_state.get("router_config")
    elif router_init_state is not None:
        saved_router_config = router_init_state.get("router_config")
    router_config = RouterConfig.from_dict(saved_router_config or requested_router_config.to_dict())
    if router_config.hidden_size != qwen.config.hidden_size or router_config.semantic_size != semantic_size:
        raise ValueError(
            "Router checkpoint dimensions do not match the current backbone/SigLIP2 encoders: "
            f"router hidden/semantic={router_config.hidden_size}/{router_config.semantic_size}, "
            f"runtime={qwen.config.hidden_size}/{semantic_size}."
        )
    router = build_router(router_config).to(device=device, dtype=dtype)
    router.train()
    if router_init_state is not None:
        router.load_state_dict(router_init_state.get("router", router_init_state), strict=True)
    if resume_state is not None:
        router.load_state_dict(resume_state["router"], strict=True)

    if world_size > 1:
        router = DistributedDataParallel(router, device_ids=[local_rank], output_device=local_rank)
        if args.training_stage == "joint":
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

    parameter_groups = [
        {"params": [parameter for parameter in router.parameters() if parameter.requires_grad], "lr": args.lr}
    ]
    lora_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.training_stage == "joint":
        if not lora_parameters:
            raise RuntimeError("No trainable LoRA parameters were created.")
        parameter_groups.append({"params": lora_parameters, "lr": args.lora_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-6)

    epoch_event_lists = [
        distributed_epoch_events(events, args.seed + epoch, rank, world_size)
        for epoch in range(args.epochs)
    ]
    scheduled_steps = sum(len(epoch_events) for epoch_events in epoch_event_lists)
    total_steps = min(args.max_steps, scheduled_steps) if args.max_steps is not None else scheduled_steps
    total_optimizer_steps = max(1, math.ceil(total_steps / args.gradient_accumulation_steps))
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_optimizer_steps)

    global_step = int(resume_state.get("step", 0)) if resume_state else 0
    optimizer_step = int(resume_state.get("optimizer_step", 0)) if resume_state else 0
    start_epoch = int(resume_state.get("next_epoch", 0)) if resume_state else 0
    start_event_index = int(resume_state.get("next_event_index", 0)) if resume_state else 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        if "python_random_state" in resume_state:
            random.setstate(resume_state["python_random_state"])
        if "torch_random_state" in resume_state:
            torch.set_rng_state(resume_state["torch_random_state"])

    trainable_router = sum(parameter.numel() for parameter in router.parameters() if parameter.requires_grad)
    trainable_lora = sum(parameter.numel() for parameter in lora_parameters)
    if rank == 0:
        print(
        json.dumps(
            {
                "training_stage": args.training_stage,
                "router_parameters": trainable_router,
                "router_config": router_config.to_dict(),
                "lora_parameters": trainable_lora if args.training_stage == "joint" else 0,
                "lora_config": None
                if args.training_stage != "joint"
                else {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "target_modules": args.lora_target_modules,
                },
                "router_lr": args.lr,
                "lora_lr": args.lora_lr if args.training_stage == "joint" else None,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "memory_aware": args.memory_aware,
                "chunkwise_memory_training": args.chunkwise_memory_training,
                "hybrid_memory_training": args.hybrid_memory_training,
                "stream_chunk_frames": args.stream_chunk_frames,
                "mixed_budget_config": None
                if args.mixed_visual_budgets is None
                else [
                    {
                        "visual_budget": visual,
                        "qim_read_budget": qim,
                        "im_read_budget": im,
                    }
                    for visual, qim, im in zip(
                        args.mixed_visual_budgets,
                        args.mixed_qim_read_budgets,
                        args.mixed_im_read_budgets,
                    )
                ],
                "memory_events": memory_event_count,
                "memory_config": {
                    "qim_capacity": args.qim_capacity,
                    "im_capacity": args.im_capacity,
                    "active_buffer_capacity": args.active_buffer_capacity,
                    "im_episode_budget": args.im_episode_budget,
                    "qim_read_budget": args.qim_read_budget,
                    "im_read_budget": args.im_read_budget,
                },
                "attn_implementation_requested": args.attn_implementation,
                "vision_attn_implementation": args.vision_attn_implementation,
                "decoder_attention_class": decoder_attention_class,
                "vision_attention_class": type(qwen.visual.blocks[0].attn).__name__,
                "cuda_sdp_backends": sdp_backends,
                "world_size": world_size,
                "events_per_rank": [len(items) for items in epoch_event_lists],
            }
        ),
        flush=True,
        )

    optimizer.zero_grad(set_to_none=True)
    final_epoch = start_epoch
    final_event_index = start_event_index
    memory_manager: Optional[ComplementaryTokenMemory] = None
    memory_dsh_state: Optional[torch.Tensor] = None
    memory_sample_id: Optional[str] = None
    memory_stream_turn_id = 0
    bad_sample_log = Path(
        args.bad_sample_log
        or (Path(args.output_dir) / f"bad_samples_rank{rank}.jsonl")
    )
    if resume_state is not None and resume_state.get("memory_runtime") is not None:
        memory_runtime = resume_state["memory_runtime"]
        memory_manager = ComplementaryTokenMemory.from_state_dict(memory_runtime["manager"])
        memory_dsh_state = memory_runtime.get("dsh_state")
        memory_sample_id = memory_runtime.get("sample_id")
        memory_stream_turn_id = int(memory_runtime.get("stream_turn_id", 0))

    def ingest_detached_history_chunk(
        event: AnswerEvent,
        video_frames: List[Image.Image],
        answer: str,
        manager: ComplementaryTokenMemory,
        dsh_state: Optional[torch.Tensor],
        temporal_offset: int,
        stream_turn_id: int,
    ) -> Tuple[Optional[torch.Tensor], int, int]:
        """Match inference ingest(): update memory without a decoder loss graph."""
        with torch.no_grad():
            chunk_inputs, _, chunk_prompt_len = build_inputs(processor, event, video_frames, answer)
            chunk_inputs = chunk_inputs.to(device)
            chunk_visual, chunk_visual_type = qwen.embedding_visual_info(**chunk_inputs)
            chunk_embeds = make_inputs_embeds(model, chunk_inputs, chunk_visual, chunk_visual_type)
            chunk_positions = (
                chunk_inputs.input_ids[0] == qwen.config.video_token_id
            ).nonzero(as_tuple=True)[0]
            chunk_hidden = chunk_embeds[:, chunk_positions]
            chunk_text, chunk_text_mask = router_text_tokens(
                chunk_embeds,
                chunk_inputs.input_ids,
                chunk_prompt_len,
                qwen.config.video_token_id,
            )
            chunk_features = encode_visual_evidence(
                video_frames=video_frames,
                video_grid_thw=chunk_inputs.video_grid_thw.detach().cpu(),
                query=current_query_text(event),
                semantic_encoder=semantic_encoder,
                semantic_preprocess=semantic_preprocess,
                semantic_processor=semantic_processor,
                semantic_device=semantic_device,
                temporal_alpha=args.temporal_alpha,
                novelty_weight=args.novelty_weight,
                fps=args.fps,
                global_temporal_offset=temporal_offset,
                initial_dsh=dsh_state,
                output_device=device,
            )
            chunk_prior = chunk_features.prior_score.unsqueeze(0).to(device=device, dtype=dtype)
            chunk_prior_features = chunk_features.prior_features.unsqueeze(0).to(
                device=device, dtype=dtype
            )
            chunk_semantic = chunk_features.semantic_keys.unsqueeze(0).to(device=device, dtype=dtype)
            raw_t, raw_h, raw_w = [
                int(value) for value in chunk_inputs.video_grid_thw[0].tolist()
            ]
            chunk_scores, _ = router(
                visual_hidden=chunk_hidden,
                text_context=chunk_text,
                prior_score=chunk_prior,
                prior_features=chunk_prior_features,
                residual_scale=min(1.0, global_step / max(args.router_residual_warmup, 1)),
                semantic_hidden=chunk_semantic,
                text_attention_mask=chunk_text_mask,
                grid_shape=(raw_t, raw_h // 2, raw_w // 2),
            )
            chunk_keep = build_budgeted_keep_mask(
                input_ids=chunk_inputs.input_ids,
                video_token_id=qwen.config.video_token_id,
                visual_scores=chunk_scores,
                keep_rate=args.keep_rate,
            )[0, chunk_positions]
            manager.observe(
                semantic_keys=chunk_features.semantic_keys,
                qwen_values=chunk_hidden[0].detach(),
                novelty=chunk_features.novelty,
                router_scores=chunk_scores[0].detach(),
                active_mask=chunk_keep.detach(),
                token_metadata=chunk_features.metadata,
                turn_id=stream_turn_id,
            )
            manager.consolidate_turn(chunk_features.query_embedding, stream_turn_id)
            return chunk_features.temporal_state, temporal_offset + raw_t, stream_turn_id + 1

    for epoch in range(start_epoch, args.epochs):
        epoch_events = epoch_event_lists[epoch]
        event_begin = start_event_index if epoch == start_epoch else 0
        if event_begin == 0:
            # Reset memory between epochs.
            memory_manager = None
            memory_dsh_state = None
            memory_sample_id = None
            memory_stream_turn_id = 0
        for event_index in range(event_begin, len(epoch_events)):
            if args.max_steps is not None and global_step >= args.max_steps:
                break
            event = epoch_events[event_index]
            (
                training_policy,
                event_visual_budget,
                event_qim_budget,
                event_im_budget,
                estimated_interval_tokens,
            ) = hybrid_event_policy(event, epoch, args)
            use_memory = bool(args.memory_aware and training_policy == "chunk_memory")
            persistent_memory = bool(args.memory_aware and event.training_mode == "memory")
            write_memory = bool(persistent_memory or use_memory)
            if write_memory and memory_sample_id != event.sample_id:
                memory_manager = ComplementaryTokenMemory(
                    qim_capacity=args.qim_capacity,
                    im_capacity=args.im_capacity,
                    active_buffer_capacity=args.active_buffer_capacity,
                    im_episode_budget=args.im_episode_budget,
                )
                memory_dsh_state = None
                memory_sample_id = event.sample_id
                memory_stream_turn_id = 0
            elif not write_memory:
                memory_manager = None
                memory_dsh_state = None
                memory_sample_id = None
                memory_stream_turn_id = 0
            global_step += 1
            final_epoch, final_event_index = _next_data_position(epoch, event_index, len(epoch_events))

            answer = target_text(event)
            history_chunks = 0
            memory_temporal_offset = max(
                0, int(round(event.visible_video_start * args.fps)) // 2
            )
            try:
                if use_memory and args.chunkwise_memory_training:
                    if memory_manager is None:
                        raise AssertionError("Chunk-wise memory runtime was not initialized.")
                    chunk_iterator = iter(iter_visible_frame_chunks(event, args))
                    try:
                        video_frames = next(chunk_iterator)
                    except StopIteration as exc:
                        raise ValueError(
                            f"No streaming chunks were produced for {event.event_id}."
                        ) from exc
                    for following_chunk in chunk_iterator:
                        memory_dsh_state, memory_temporal_offset, memory_stream_turn_id = (
                            ingest_detached_history_chunk(
                                event=event,
                                video_frames=video_frames,
                                answer=answer,
                                manager=memory_manager,
                                dsh_state=memory_dsh_state,
                                temporal_offset=memory_temporal_offset,
                                stream_turn_id=memory_stream_turn_id,
                            )
                        )
                        history_chunks += 1
                        video_frames = following_chunk
                else:
                    video_frames = sample_visible_frames(
                        event,
                        args,
                        force_causal_prefix=event.training_mode == "memory" and not args.memory_aware,
                        visual_token_budget=event_visual_budget,
                    )
            except (OSError, ValueError, av.error.FFmpegError) as exc:
                if not args.skip_bad_samples:
                    raise
                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "event_index": event_index,
                    "event_id": event.event_id,
                    "sample_id": event.sample_id,
                    "video": event.video,
                    "visible_span": [event.visible_video_start, event.visible_video_end],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                bad_sample_log.parent.mkdir(parents=True, exist_ok=True)
                with bad_sample_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(json.dumps({"skipped_bad_sample": record}, ensure_ascii=False), flush=True)
                skip_boundary = (
                    global_step % args.gradient_accumulation_steps == 0
                    or global_step >= total_steps
                )
                if skip_boundary:
                    trainable_parameters = [
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                        if parameter.grad is not None
                    ]
                    if trainable_parameters:
                        if args.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                trainable_parameters, args.max_grad_norm
                            )
                        optimizer.step()
                        scheduler.step()
                        optimizer_step += 1
                    optimizer.zero_grad(set_to_none=True)
                continue
            inputs, labels, prompt_len = build_inputs(processor, event, video_frames, answer)
            inputs = inputs.to(device)
            labels = labels.to(device)

            visual_embedded, visual_embedded_type = qwen.embedding_visual_info(**inputs)
            inputs_embeds = make_inputs_embeds(model, inputs, visual_embedded, visual_embedded_type)
            video_positions = (inputs.input_ids[0] == qwen.config.video_token_id).nonzero(as_tuple=True)[0]
            visual_hidden = inputs_embeds[:, video_positions]
            text_context, text_attention_mask = router_text_tokens(
                inputs_embeds, inputs.input_ids, prompt_len, qwen.config.video_token_id
            )

            memory_features = None
            if write_memory:
                memory_features = encode_visual_evidence(
                    video_frames=video_frames,
                    video_grid_thw=inputs.video_grid_thw.detach().cpu(),
                    query=current_query_text(event),
                    semantic_encoder=semantic_encoder,
                    semantic_preprocess=semantic_preprocess,
                    semantic_processor=semantic_processor,
                    semantic_device=semantic_device,
                    temporal_alpha=args.temporal_alpha,
                    novelty_weight=args.novelty_weight,
                    fps=args.fps,
                    global_temporal_offset=memory_temporal_offset,
                    initial_dsh=memory_dsh_state,
                    output_device=device,
                )
                memory_dsh_state = memory_features.temporal_state
                prior_score = memory_features.prior_score
                prior_features = memory_features.prior_features
                semantic_hidden = memory_features.semantic_keys
            else:
                dialogue_context_text = "\n".join(
                    f"{message['role']}: {message['text']}" for message in event.dialogue_context
                )
                prior_score, prior_features, semantic_hidden = build_continuous_prior(
                    video_frames=video_frames,
                    video_grid_thw=inputs.video_grid_thw.detach().cpu(),
                    dialogue_context_text=dialogue_context_text,
                    semantic_encoder=semantic_encoder,
                    semantic_preprocess=semantic_preprocess,
                    semantic_processor=semantic_processor,
                    semantic_device=semantic_device,
                    temporal_alpha=args.temporal_alpha,
                    novelty_weight=args.novelty_weight,
                )
            prior_score = prior_score.unsqueeze(0).to(device=device, dtype=dtype)
            prior_features = prior_features.unsqueeze(0).to(device=device, dtype=dtype)
            semantic_hidden = semantic_hidden.unsqueeze(0).to(device=device, dtype=dtype)
            raw_grid_t, raw_grid_h, raw_grid_w = [int(value) for value in inputs.video_grid_thw[0].tolist()]
            grid_shape = (raw_grid_t, raw_grid_h // 2, raw_grid_w // 2)

            residual_scale = min(1.0, global_step / max(args.router_residual_warmup, 1))
            refined_score, _ = router(
                visual_hidden=visual_hidden,
                text_context=text_context,
                prior_score=prior_score,
                prior_features=prior_features,
                residual_scale=residual_scale,
                semantic_hidden=semantic_hidden,
                text_attention_mask=text_attention_mask,
                grid_shape=grid_shape,
            )
            router_bias = build_router_attention_bias(
                input_ids=inputs.input_ids,
                video_token_id=qwen.config.video_token_id,
                visual_scores=refined_score,
            )
            router_keep_mask = build_budgeted_keep_mask(
                input_ids=inputs.input_ids,
                video_token_id=qwen.config.video_token_id,
                visual_scores=refined_score,
                keep_rate=args.keep_rate,
            )

            prior_reg_w = args.prior_reg_end + (args.prior_reg_start - args.prior_reg_end) * max(
                0.0, 1.0 - global_step / max(total_steps, 1)
            )
            recalled_qim = 0
            recalled_im = 0
            # Commit detached short-turn state before later memory retrieval.
            if write_memory and not use_memory:
                if memory_manager is None or memory_features is None:
                    raise AssertionError("Hybrid memory runtime was not initialized.")
                current_keep_mask = router_keep_mask[0, video_positions]
                current_memory_turn = memory_stream_turn_id
                memory_manager.observe(
                    semantic_keys=memory_features.semantic_keys,
                    qwen_values=visual_hidden[0].detach(),
                    novelty=memory_features.novelty,
                    router_scores=refined_score[0].detach(),
                    active_mask=current_keep_mask.detach(),
                    token_metadata=memory_features.metadata,
                    turn_id=current_memory_turn,
                )
                memory_manager.consolidate_turn(memory_features.query_embedding, current_memory_turn)
                memory_stream_turn_id += 1
            if use_memory:
                if memory_manager is None or memory_features is None:
                    raise AssertionError("Memory-aware runtime was not initialized.")
                current_keep_mask = router_keep_mask[0, video_positions]
                current_token_count = int(current_keep_mask.sum().item())
                remaining_read_budget = max(0, event_visual_budget - current_token_count)
                effective_qim_budget = min(event_qim_budget, remaining_read_budget)
                effective_im_budget = min(
                    event_im_budget,
                    max(0, remaining_read_budget - effective_qim_budget),
                )
                recalled = memory_manager.retrieve(
                    query_embedding=memory_features.query_embedding,
                    qim_budget=effective_qim_budget,
                    im_budget=effective_im_budget,
                )
                current_memory_turn = (
                    memory_stream_turn_id if args.chunkwise_memory_training else event.turn_index
                )
                memory_manager.observe(
                    semantic_keys=memory_features.semantic_keys,
                    qwen_values=visual_hidden[0].detach(),
                    novelty=memory_features.novelty,
                    router_scores=refined_score[0].detach(),
                    active_mask=current_keep_mask.detach(),
                    token_metadata=memory_features.metadata,
                    turn_id=current_memory_turn,
                )
                packed = pack_visual_context(
                    model=qwen,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    current_visual_values=visual_hidden[0],
                    current_keep_mask=current_keep_mask,
                    current_metadata=memory_features.metadata,
                    recalled_slots=recalled.slots,
                )
                packed_labels = packed_training_labels(
                    original_input_ids=inputs.input_ids,
                    original_labels=labels,
                    packed_input_ids=packed.input_ids,
                    video_token_id=qwen.config.video_token_id,
                )
                outputs = model(
                    input_ids=packed.input_ids,
                    inputs_embeds=packed.inputs_embeds,
                    attention_mask=packed.attention_mask,
                    position_ids=packed.position_ids,
                    labels=packed_labels,
                    use_cache=False,
                )
                memory_manager.consolidate_turn(memory_features.query_embedding, current_memory_turn)
                if args.chunkwise_memory_training:
                    memory_stream_turn_id += 1
                recalled_qim = len(recalled.qim_slots)
                recalled_im = len(recalled.im_slots)
            else:
                outputs = model(
                    **inputs,
                    inputs_embeds=inputs_embeds,
                    visual_embedded=None,
                    visual_embedded_type=None,
                    labels=labels,
                    use_cache=False,
                    router_attention_bias=router_bias,
                    router_attention_layer=args.router_layer,
                    router_prune_keep_mask=router_keep_mask,
                    router_prune_after_layer=args.prune_after_layer,
                    print_token_stats=args.print_token_stats,
                )
            prior_loss = F.mse_loss(refined_score, prior_score)
            loss = outputs.loss + prior_reg_w * prior_loss
            (loss / args.gradient_accumulation_steps).backward()

            should_update = (
                global_step % args.gradient_accumulation_steps == 0
                or global_step >= total_steps
            )
            if should_update:
                trainable_parameters = [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                    if parameter.grad is not None
                ]
                if args.max_grad_norm > 0 and trainable_parameters:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            if rank == 0 and (global_step <= 2 or global_step % 10 == 0):
                print(
                    json.dumps(
                        {
                            "step": global_step,
                            "optimizer_step": optimizer_step,
                            "epoch": epoch,
                            "event_id": event.event_id,
                            "sample_id": event.sample_id,
                            "training_mode": "memory" if use_memory else "causal_prefix",
                            "training_policy": training_policy,
                            "turn_index": event.turn_index,
                            "visual_budget": event_visual_budget,
                            "estimated_interval_visual_tokens": estimated_interval_tokens,
                            "history_chunks": history_chunks,
                            "answer_chunk_frames": len(video_frames),
                            "loss": float(loss.detach().cpu()),
                            "answer_loss": float(outputs.loss.detach().cpu()),
                            "prior_loss": float(prior_loss.detach().cpu()),
                            "prior_reg_weight": prior_reg_w,
                            "keep_rate": args.keep_rate,
                            "recalled_qim_tokens": recalled_qim,
                            "recalled_im_tokens": recalled_im,
                            "router_lr": optimizer.param_groups[0]["lr"],
                            "lora_lr": optimizer.param_groups[1]["lr"] if args.training_stage == "joint" else None,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            # Release the completed activation graph before the next sample.
            del outputs, loss, prior_loss, router_bias, router_keep_mask
            del refined_score, visual_hidden, inputs_embeds, visual_embedded
            del inputs, labels, semantic_hidden, prior_score, prior_features
            if rank == 0 and should_update and args.save_every > 0 and global_step % args.save_every == 0:
                save_training_checkpoint(
                    router,
                    model,
                    optimizer,
                    scheduler,
                    args,
                    global_step,
                    optimizer_step,
                    final_epoch,
                    final_event_index,
                    memory_manager,
                    memory_dsh_state,
                    memory_sample_id,
                    memory_stream_turn_id,
                )

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        final_path = save_training_checkpoint(
            router,
            model,
            optimizer,
            scheduler,
            args,
            global_step,
            optimizer_step,
            final_epoch,
            final_event_index,
            memory_manager,
            memory_dsh_state,
            memory_sample_id,
            memory_stream_turn_id,
        )
        print(f"Saved training checkpoint to {final_path}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
