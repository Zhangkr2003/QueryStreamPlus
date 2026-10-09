import importlib.util
import math
import os
from typing import List, Tuple

import av
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TVF


DEFAULT_LOCAL_SIGLIP2_CHECKPOINT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "siglip2-large-patch16-256")
)


def resolve_attention_backend(requested: str, device: str | torch.device = "cuda") -> str:
    """Resolve the requested attention backend."""
    if requested not in {"auto", "eager", "sdpa", "flash_attention_2"}:
        raise ValueError(f"Unsupported attention implementation: {requested}")
    if requested != "auto":
        return requested
    device_type = torch.device(device).type
    has_flash_attention = importlib.util.find_spec("flash_attn") is not None
    return "flash_attention_2" if device_type == "cuda" and has_flash_attention else "sdpa"


def _normalize_device_map(device_map: str):
    if device_map.lower() == "none":
        return None
    if device_map.lower() == "cpu":
        return {"": "cpu"}
    return device_map


def _first_real_device(model) -> torch.device:
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.device.type != "meta":
            return tensor.device
    return torch.device("cpu")


def round_by_factor(number, factor):
    return round(number / factor) * factor


def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def floor_by_factor(number, factor):
    return math.floor(number / factor) * factor


def smart_resize(height, width, factor=28, min_pixels=128 * 28 * 28, max_pixels=128 * 28 * 28):
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return int(h_bar), int(w_bar)


def load_video_frames(
    video_path,
    fps,
    min_frames,
    max_frames,
    min_pixels,
    max_pixels,
    end_time=None,
    return_timestamps=False,
):
    """Decode uniformly sampled, resized frames."""
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate) if stream.average_rate else 1.0
        declared_frames = int(stream.frames or 0)
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        elif declared_frames > 0:
            duration = declared_frames / source_fps
        else:
            duration = 0.0

        if end_time is not None:
            if float(end_time) <= 0:
                raise ValueError("end_time must be positive when provided.")
            duration = min(duration, float(end_time)) if duration > 0 else float(end_time)
        estimated_total = max(1, int(round(duration * source_fps))) if duration > 0 else declared_frames
        target_frames = (duration if duration > 0 else estimated_total / source_fps) * fps
        upper_bound = max_frames
        if estimated_total > 0:
            upper_bound = min(upper_bound, estimated_total)
        target_frames = min(max(target_frames, min_frames), upper_bound)
        target_frames = max(2, round_by_factor(target_frames, 2))
        if estimated_total > 0:
            target_frames = min(target_frames, estimated_total)
        target_frames = max(1, int(target_frames))

        if duration > 0:
            end_time = max(0.0, duration - 1.0 / max(source_fps, 1e-6))
            target_times = torch.linspace(0.0, end_time, target_frames).tolist()
        else:
            # Fall back to bounded chronological sampling.
            target_times = None

        sampled = []
        sampled_timestamps = []
        target_index = 0
        next_fallback_time = 0.0
        first_timestamp = None
        last_frame = None
        last_timestamp = None
        resized_size = None

        def materialize(frame):
            nonlocal resized_size
            image = frame.to_image().convert("RGB")
            if resized_size is None:
                width, height = image.size
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                resized_size = (resized_width, resized_height)
            return image.resize(resized_size, Image.Resampling.BICUBIC)

        for decode_index, frame in enumerate(container.decode(stream)):
            last_frame = frame
            timestamp = float(frame.time) if frame.time is not None else decode_index / source_fps
            if first_timestamp is None:
                first_timestamp = timestamp
            timestamp -= first_timestamp
            last_timestamp = timestamp
            if target_times is None:
                if len(sampled) >= max_frames or timestamp + 1e-9 < next_fallback_time:
                    continue
                sampled.append(materialize(frame))
                sampled_timestamps.append(timestamp)
                next_fallback_time += 1.0 / max(fps, 1e-6)
                continue
            if target_index >= len(target_times):
                break
            # Use the first decoded frame at or after each target time.
            if timestamp + 1e-9 < target_times[target_index]:
                continue
            image = materialize(frame)
            first_use = True
            while target_index < len(target_times) and timestamp + 1e-9 >= target_times[target_index]:
                sampled.append(image if first_use else image.copy())
                sampled_timestamps.append(timestamp)
                first_use = False
                target_index += 1

        if target_times is not None and target_index < len(target_times) and last_frame is not None:
            final_image = materialize(last_frame)
            while target_index < len(target_times):
                sampled.append(final_image.copy())
                sampled_timestamps.append(float(last_timestamp or 0.0))
                target_index += 1
    finally:
        container.close()
    if not sampled:
        raise ValueError(f"No video frames could be decoded from {video_path}")

    while len(sampled) < min_frames:
        sampled.append(sampled[-1].copy())
        sampled_timestamps.append(sampled_timestamps[-1])
    if len(sampled) % 2 == 1:
        sampled.append(sampled[-1].copy())
        sampled_timestamps.append(sampled_timestamps[-1])
    if return_timestamps:
        return sampled, sampled_timestamps
    return sampled


DEFAULT_SIGLIP2_MODEL = "google/siglip2-large-patch16-256"


class _Siglip2ProcessorAdapter:
    """Wrap SigLIP2 image and text processors."""

    backend = "siglip2"

    def __init__(self, image_processor, tokenizer=None):
        self.image_processor = image_processor
        self.tokenizer = tokenizer if tokenizer is not None else image_processor


def _load_semantic_encoder(
    model_name: str,
    checkpoint: str | None,
    device: torch.device,
):
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    checkpoint = checkpoint or model_name
    model = AutoModel.from_pretrained(checkpoint).to(device).eval()
    # Use the fast tokenizer without requiring SentencePiece.
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    adapter = _Siglip2ProcessorAdapter(image_processor, tokenizer)
    return model, adapter, adapter


def _add_semantic_encoder_args(parser) -> None:
    parser.add_argument("--siglip2-model", default=DEFAULT_SIGLIP2_MODEL)
    parser.add_argument(
        "--siglip2-checkpoint",
        default=None,
        help="Local SigLIP2 directory. Public inference defaults to models/siglip2-large-patch16-256.",
    )


def _semantic_encoder_spec(args):
    return args.siglip2_model, args.siglip2_checkpoint


@torch.no_grad()
def _siglip2_patch_features(model, images: dict[str, torch.Tensor]) -> torch.Tensor:
    vision_kwargs = {"pixel_values": images["pixel_values"]}
    if "pixel_attention_mask" in images:
        vision_kwargs["attention_mask"] = images["pixel_attention_mask"]
    return model.vision_model(**vision_kwargs).last_hidden_state


@torch.no_grad()
def _siglip2_text_feature(model, processor, query: str, device: torch.device) -> torch.Tensor:
    text = processor.tokenizer(
        [query],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    text = {key: value.to(device) for key, value in text.items()}
    text_feat = model.get_text_features(**text)
    return F.normalize(text_feat, dim=-1)[0]


def _video_input_to_pil_frames(video_input) -> List[Image.Image]:
    if isinstance(video_input, (list, tuple)):
        return [frame.convert("RGB") if isinstance(frame, Image.Image) else TVF.to_pil_image(frame) for frame in video_input]

    frames = []
    for frame in video_input:
        frame = frame.detach().cpu().clamp(0, 255).to(torch.uint8)
        frames.append(TVF.to_pil_image(frame))
    return frames


def _prepare_siglip2_batch(frames: List[Image.Image], processor, device: torch.device):
    batch = processor.image_processor(images=frames, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items()}


def _overlap_weights(
    source_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    weights = torch.zeros(target_h * target_w, source_h * source_w, device=device, dtype=dtype)

    for tr in range(target_h):
        ty0 = tr / target_h
        ty1 = (tr + 1) / target_h
        for tc in range(target_w):
            tx0 = tc / target_w
            tx1 = (tc + 1) / target_w
            target_idx = tr * target_w + tc
            for sr in range(source_h):
                sy0 = sr / source_h
                sy1 = (sr + 1) / source_h
                y_overlap = max(0.0, min(ty1, sy1) - max(ty0, sy0))
                if y_overlap == 0.0:
                    continue
                for sc in range(source_w):
                    sx0 = sc / source_w
                    sx1 = (sc + 1) / source_w
                    x_overlap = max(0.0, min(tx1, sx1) - max(tx0, sx0))
                    if x_overlap == 0.0:
                        continue
                    source_idx = sr * source_w + sc
                    weights[target_idx, source_idx] = y_overlap * x_overlap

    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weights


def _pool_siglip2_features_to_qwen_grid(
    semantic_features: torch.Tensor,
    qwen_grid_hw: Tuple[int, int],
) -> torch.Tensor:
    num_patches = semantic_features.shape[0]
    semantic_grid = int(math.sqrt(num_patches))
    if semantic_grid * semantic_grid != num_patches:
        raise ValueError(f"SigLIP2 patch count must form a square grid, got {num_patches}.")

    weights = _overlap_weights(
        source_hw=(semantic_grid, semantic_grid),
        target_hw=qwen_grid_hw,
        device=semantic_features.device,
        dtype=semantic_features.dtype,
    )
    return weights @ semantic_features
