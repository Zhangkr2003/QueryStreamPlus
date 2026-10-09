"""Run QueryStream inference on a local video."""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import av
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TVF
from transformers import AutoProcessor

from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


DEFAULT_LOCAL_CLIP_PRETRAINED = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "openclip", "ViT-L-14-openai.safetensors")
)


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


@dataclass
class QueryStreamStats:
    keep_rate: float
    per_frame_keep_rates: List[float]
    per_step_total_tokens: List[int]
    per_step_kept_tokens: List[int]
    per_step_dropped_tokens: List[int]
    relevance_scores: List[float]


def print_qdp_token_stats(stats: QueryStreamStats, fps: float, label: str = "QDP") -> None:
    print(f"[{label} per-step token stats]", file=sys.stderr, flush=True)
    for step, (total, kept, dropped, keep_rate, relevance) in enumerate(
        zip(
            stats.per_step_total_tokens,
            stats.per_step_kept_tokens,
            stats.per_step_dropped_tokens,
            stats.per_frame_keep_rates,
            stats.relevance_scores,
        )
    ):
        frame_begin = step * 2
        frame_end = frame_begin + 1
        time_end = (step + 1) * 2.0 / fps
        print(
            f"  step={step:03d} sampled_frames=[{frame_begin},{frame_end}] "
            f"time_end≈{time_end:.3f}s tokens={total} kept={kept} dropped={dropped} "
            f"keep_rate={keep_rate:.4f} drop_rate={1.0 - keep_rate:.4f} "
            f"query_relevance={relevance:.4f}",
            file=sys.stderr,
            flush=True,
        )
    total = sum(stats.per_step_total_tokens)
    kept = sum(stats.per_step_kept_tokens)
    dropped = sum(stats.per_step_dropped_tokens)
    print(
        f"[{label} total visual tokens] before={total} after={kept} dropped={dropped} "
        f"keep_rate={kept / max(total, 1):.4f} drop_rate={dropped / max(total, 1):.4f}",
        file=sys.stderr,
        flush=True,
    )


@dataclass
class QueryStreamVisualMasks:
    masks: List[torch.Tensor]
    qwen_grid_hw: Tuple[int, int]
    stats: QueryStreamStats


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


def load_video_frames(video_path, fps, min_frames, max_frames, min_pixels, max_pixels):
    container = av.open(video_path)
    stream = container.streams.video[0]
    source_fps = float(stream.average_rate) if stream.average_rate else 1.0
    frames = [frame.to_image().convert("RGB") for frame in container.decode(stream)]
    container.close()
    if not frames:
        raise ValueError(f"No video frames could be decoded from {video_path}")

    total_frames = len(frames)
    target_frames = total_frames / source_fps * fps
    target_frames = min(max(target_frames, min_frames), min(max_frames, total_frames))
    target_frames = max(2, round_by_factor(target_frames, 2))
    target_frames = min(target_frames, total_frames)
    indices = torch.linspace(0, total_frames - 1, int(target_frames)).round().long().tolist()
    sampled = [frames[i] for i in indices]
    while len(sampled) < min_frames:
        sampled.append(sampled[-1].copy())
    if len(sampled) % 2 == 1:
        sampled.append(sampled[-1].copy())

    width, height = sampled[0].size
    resized_height, resized_width = smart_resize(
        height,
        width,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return [
        frame.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
        for frame in sampled
    ]


def _load_openclip(model_name: str, pretrained: str, device: torch.device):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "QueryStream requires open_clip_torch for query-aware patch features. "
            "Install it in the TimeChat-Online environment with: pip install open_clip_torch"
        ) from exc

    load_kwargs = {}
    if os.path.isfile(pretrained):
        load_kwargs["weights_only"] = False
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        **load_kwargs,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, preprocess, tokenizer


def _manual_openclip_vit_tokens(visual, images: torch.Tensor) -> torch.Tensor:
    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype)
    cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.patch_dropout(x) if hasattr(visual, "patch_dropout") else x
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    if hasattr(visual, "ln_post"):
        x = visual.ln_post(x)
    return x[:, 1:, :]


@torch.no_grad()
def _openclip_patch_features(model, images: torch.Tensor) -> torch.Tensor:
    visual = model.visual

    def project_features(feats: torch.Tensor) -> torch.Tensor:
        proj = getattr(visual, "proj", None)
        if proj is not None and feats.shape[-1] == proj.shape[0]:
            feats = feats @ proj.to(dtype=feats.dtype, device=feats.device)
        return feats

    if hasattr(visual, "forward_features"):
        feats = visual.forward_features(images)
        if isinstance(feats, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "tokens", "x"):
                if key in feats:
                    feats = feats[key]
                    break
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim == 3:
            if feats.shape[1] == getattr(visual, "grid_size", (0, 0))[0] * getattr(visual, "grid_size", (0, 0))[1] + 1:
                feats = feats[:, 1:, :]
            return project_features(feats)

    if hasattr(visual, "trunk") and hasattr(visual.trunk, "forward_features"):
        feats = visual.trunk.forward_features(images)
        if isinstance(feats, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "tokens", "x"):
                if key in feats:
                    feats = feats[key]
                    break
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim == 4:
            feats = feats.flatten(2).transpose(1, 2)
        if feats.ndim == 3:
            if feats.shape[1] > 1 and int(math.sqrt(feats.shape[1] - 1)) ** 2 == feats.shape[1] - 1:
                feats = feats[:, 1:, :]
            return project_features(feats)

    required = ("conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer")
    if all(hasattr(visual, name) for name in required):
        return project_features(_manual_openclip_vit_tokens(visual, images))

    raise RuntimeError(
        "Could not extract patch tokens from this OpenCLIP visual encoder. "
        "Try --clip-model ViT-L-14 with an open_clip_torch VisionTransformer checkpoint."
    )


@torch.no_grad()
def _openclip_text_feature(model, tokenizer, query: str, device: torch.device) -> torch.Tensor:
    text = tokenizer([query]).to(device)
    text_feat = model.encode_text(text)
    return F.normalize(text_feat, dim=-1)[0]


def _video_input_to_pil_frames(video_input) -> List[Image.Image]:
    if isinstance(video_input, (list, tuple)):
        return [frame.convert("RGB") if isinstance(frame, Image.Image) else TVF.to_pil_image(frame) for frame in video_input]

    frames = []
    for frame in video_input:
        frame = frame.detach().cpu().clamp(0, 255).to(torch.uint8)
        frames.append(TVF.to_pil_image(frame))
    return frames


def _prepare_clip_batch(frames: List[Image.Image], preprocess, device: torch.device) -> torch.Tensor:
    return torch.stack([preprocess(frame) for frame in frames], dim=0).to(device)


def _crop_qwen_token_regions(frame: Image.Image, qwen_grid_hw: Tuple[int, int]) -> List[Image.Image]:
    grid_h, grid_w = qwen_grid_hw
    width, height = frame.size
    regions = []
    for row in range(grid_h):
        top = round(row * height / grid_h)
        bottom = round((row + 1) * height / grid_h)
        for col in range(grid_w):
            left = round(col * width / grid_w)
            right = round((col + 1) * width / grid_w)
            regions.append(frame.crop((left, top, max(right, left + 1), max(bottom, top + 1))).convert("RGB"))
    return regions


@torch.no_grad()
def _openclip_image_features(
    model,
    images: List[Image.Image],
    preprocess,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    features = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        batch = _prepare_clip_batch(batch_images, preprocess, device)
        image_features = model.encode_image(batch)
        features.append(image_features.detach())
    return torch.cat(features, dim=0)


def _resize_mask(mask: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    mask_f = mask.float().view(1, 1, mask.shape[0], mask.shape[1])
    resized = F.interpolate(mask_f, size=out_hw, mode="nearest")
    return resized.view(out_hw).bool()


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


def _pool_clip_features_to_qwen_grid(
    clip_features: torch.Tensor,
    qwen_grid_hw: Tuple[int, int],
) -> torch.Tensor:
    num_patches = clip_features.shape[0]
    clip_grid = int(math.sqrt(num_patches))
    if clip_grid * clip_grid != num_patches:
        raise ValueError(f"OpenCLIP patch count must form a square grid, got {num_patches}.")

    weights = _overlap_weights(
        source_hw=(clip_grid, clip_grid),
        target_hw=qwen_grid_hw,
        device=clip_features.device,
        dtype=clip_features.dtype,
    )
    return weights @ clip_features


def _make_qdp_masks_from_aligned_features(
    aligned_features: torch.Tensor,
    query_feature: torch.Tensor,
    grid_t: int,
    alpha: float,
    tau_temp: float,
) -> Tuple[List[torch.Tensor], List[float], List[float]]:
    aligned_features = F.normalize(aligned_features, dim=-1)
    frame_indices = [min(i * 2 + 1, aligned_features.shape[0] - 1) for i in range(grid_t)]
    dsh = None
    masks: List[torch.Tensor] = []
    keep_rates: List[float] = []
    relevance_scores: List[float] = []

    for temporal_idx, frame_idx in enumerate(frame_indices):
        feats = aligned_features[frame_idx]
        sim_query = feats @ query_feature
        semantic = sim_query > sim_query.mean()
        relevance_scores.append(float((F.normalize(feats.mean(dim=0), dim=0) @ query_feature).item()))

        if dsh is None:
            qdp = torch.ones_like(semantic, dtype=torch.bool)
            dsh = feats.clone()
        else:
            novelty = (feats * F.normalize(dsh, dim=-1)).sum(dim=-1) < tau_temp
            qdp = semantic & novelty
            dsh = alpha * feats + (1.0 - alpha) * dsh

        masks.append(qdp.detach().cpu())
        keep_rates.append(float(qdp.float().mean().item()))

    return masks, keep_rates, relevance_scores


def _make_qdp_masks(
    patch_features: torch.Tensor,
    query_feature: torch.Tensor,
    grid_t: int,
    qwen_grid_hw: Tuple[int, int],
    alpha: float,
    tau_temp: float,
) -> Tuple[List[torch.Tensor], List[float], List[float]]:
    patch_features = F.normalize(patch_features, dim=-1)
    num_frames, num_patches, _ = patch_features.shape
    clip_grid = int(math.sqrt(num_patches))
    if clip_grid * clip_grid != num_patches:
        raise ValueError(f"OpenCLIP patch count must form a square grid, got {num_patches}.")

    # Qwen2.5-VL packs two frames into one temporal visual token. Match TimeChat's
    # DTD convention by using the last sampled frame of each temporal patch.
    frame_indices = [min(i * 2 + 1, num_frames - 1) for i in range(grid_t)]
    dsh = None
    masks: List[torch.Tensor] = []
    keep_rates: List[float] = []
    relevance_scores: List[float] = []

    for temporal_idx, frame_idx in enumerate(frame_indices):
        feats = patch_features[frame_idx]
        sim_query = feats @ query_feature
        semantic = sim_query > sim_query.mean()
        relevance_scores.append(float((F.normalize(feats.mean(dim=0), dim=0) @ query_feature).item()))

        if dsh is None:
            qdp = torch.ones_like(semantic, dtype=torch.bool)
            dsh = feats.clone()
        else:
            novelty = (feats * F.normalize(dsh, dim=-1)).sum(dim=-1) < tau_temp
            qdp = semantic & novelty
            dsh = alpha * feats + (1.0 - alpha) * dsh

        qdp_grid = qdp.view(clip_grid, clip_grid)
        qwen_mask = _resize_mask(qdp_grid, qwen_grid_hw)
        masks.append(qwen_mask.flatten())
        keep_rates.append(float(qwen_mask.float().mean().item()))

    return masks, keep_rates, relevance_scores


def _make_overlap_qdp_masks(
    patch_features: torch.Tensor,
    query_feature: torch.Tensor,
    grid_t: int,
    qwen_grid_hw: Tuple[int, int],
    alpha: float,
    tau_temp: float,
) -> Tuple[List[torch.Tensor], List[float], List[float]]:
    patch_features = F.normalize(patch_features, dim=-1)
    pooled = torch.stack(
        [_pool_clip_features_to_qwen_grid(frame_feats, qwen_grid_hw) for frame_feats in patch_features],
        dim=0,
    )
    return _make_qdp_masks_from_aligned_features(
        aligned_features=pooled,
        query_feature=query_feature,
        grid_t=grid_t,
        alpha=alpha,
        tau_temp=tau_temp,
    )


def _make_qwen_region_qdp_masks(
    frames: List[Image.Image],
    query_feature: torch.Tensor,
    clip_model,
    clip_preprocess,
    clip_device: torch.device,
    grid_t: int,
    qwen_grid_hw: Tuple[int, int],
    alpha: float,
    tau_temp: float,
    clip_batch_size: int,
) -> Tuple[List[torch.Tensor], List[float], List[float]]:
    # Qwen2.5-VL packs two frames into one temporal visual step. Use the
    # latter frame in each pair, then score the exact Qwen token regions.
    frame_indices = [min(i * 2 + 1, len(frames) - 1) for i in range(grid_t)]
    dsh = None
    masks: List[torch.Tensor] = []
    keep_rates: List[float] = []
    relevance_scores: List[float] = []

    for temporal_idx, frame_idx in enumerate(frame_indices):
        qwen_regions = _crop_qwen_token_regions(frames[frame_idx], qwen_grid_hw)
        feats = _openclip_image_features(
            clip_model,
            qwen_regions,
            clip_preprocess,
            clip_device,
            batch_size=clip_batch_size,
        )
        feats = F.normalize(feats, dim=-1)
        sim_query = feats @ query_feature
        semantic = sim_query > sim_query.mean()
        relevance_scores.append(float((F.normalize(feats.mean(dim=0), dim=0) @ query_feature).item()))

        if dsh is None:
            qdp = torch.ones_like(semantic, dtype=torch.bool)
            dsh = feats.clone()
        else:
            novelty = (feats * F.normalize(dsh, dim=-1)).sum(dim=-1) < tau_temp
            qdp = semantic & novelty
            dsh = alpha * feats + (1.0 - alpha) * dsh

        masks.append(qdp.detach().cpu())
        keep_rates.append(float(qdp.float().mean().item()))

    return masks, keep_rates, relevance_scores


def build_querystream_keep_mask(
    video_tensor: torch.Tensor,
    video_grid_thw: torch.Tensor,
    input_ids: torch.Tensor,
    vision_start_token_id: int,
    vision_end_token_id: int,
    query: str,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    clip_device: torch.device,
    alpha: float = 0.1,
    tau_temp: float = 0.90,
    clip_batch_size: int = 64,
    qdp_align_method: str = "overlap",
) -> Tuple[torch.Tensor, QueryStreamStats]:
    if video_grid_thw is None or len(video_grid_thw) == 0:
        raise ValueError("QueryStream CLI currently expects exactly one video input.")

    visual_masks = build_querystream_visual_masks(
        video_tensor=video_tensor,
        video_grid_thw=video_grid_thw,
        query=query,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer,
        clip_device=clip_device,
        alpha=alpha,
        tau_temp=tau_temp,
        clip_batch_size=clip_batch_size,
        qdp_align_method=qdp_align_method,
    )
    visual_keep = torch.cat(visual_masks.masks, dim=0).to(video_grid_thw.device)
    sample_input_ids = input_ids[0]
    keep_mask = torch.ones(sample_input_ids.shape[0], dtype=torch.bool, device=sample_input_ids.device)
    vision_start_indices = (sample_input_ids == vision_start_token_id).nonzero(as_tuple=True)[0]
    vision_end_indices = (sample_input_ids == vision_end_token_id).nonzero(as_tuple=True)[0]
    if len(vision_start_indices) != 1 or len(vision_end_indices) != 1:
        raise ValueError("QueryStream CLI currently expects exactly one video span in the prompt.")
    visual_indices = torch.arange(
        vision_start_indices[0] + 1,
        vision_end_indices[0],
        device=sample_input_ids.device,
    )
    if visual_indices.numel() != visual_keep.numel():
        raise ValueError(
            "QueryStream mask/token mismatch: "
            f"mask has {visual_keep.numel()} visual entries, prompt has {visual_indices.numel()} visual tokens."
        )
    keep_mask[visual_indices] = visual_keep

    return keep_mask, visual_masks.stats


def build_querystream_visual_masks(
    video_tensor,
    video_grid_thw: torch.Tensor,
    query: str,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    clip_device: torch.device,
    alpha: float = 0.1,
    tau_temp: float = 0.90,
    clip_batch_size: int = 64,
    qdp_align_method: str = "overlap",
) -> QueryStreamVisualMasks:
    if video_grid_thw is None or len(video_grid_thw) == 0:
        raise ValueError("QueryStream visualization expects exactly one video input.")

    grid_t, grid_h, grid_w = [int(x) for x in video_grid_thw[0].tolist()]
    qwen_h = grid_h // 2
    qwen_w = grid_w // 2
    frames = _video_input_to_pil_frames(video_tensor)
    query_feature = _openclip_text_feature(clip_model, clip_tokenizer, query, clip_device)

    if qdp_align_method == "crop":
        qdp_masks, keep_rates, relevance_scores = _make_qwen_region_qdp_masks(
            frames=frames,
            query_feature=query_feature,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            clip_device=clip_device,
            grid_t=grid_t,
            qwen_grid_hw=(qwen_h, qwen_w),
            alpha=alpha,
            tau_temp=tau_temp,
            clip_batch_size=clip_batch_size,
        )
    elif qdp_align_method in {"nearest", "overlap"}:
        clip_images = _prepare_clip_batch(frames, clip_preprocess, clip_device)
        patch_features = _openclip_patch_features(clip_model, clip_images)
        if qdp_align_method == "nearest":
            qdp_masks, keep_rates, relevance_scores = _make_qdp_masks(
                patch_features=patch_features,
                query_feature=query_feature,
                grid_t=grid_t,
                qwen_grid_hw=(qwen_h, qwen_w),
                alpha=alpha,
                tau_temp=tau_temp,
            )
        else:
            qdp_masks, keep_rates, relevance_scores = _make_overlap_qdp_masks(
                patch_features=patch_features,
                query_feature=query_feature,
                grid_t=grid_t,
                qwen_grid_hw=(qwen_h, qwen_w),
                alpha=alpha,
                tau_temp=tau_temp,
            )
    else:
        raise ValueError(
            f"Unsupported qdp_align_method={qdp_align_method!r}; "
            "choose from 'overlap', 'nearest', or 'crop'."
        )

    visual_keep = torch.cat(qdp_masks, dim=0)
    per_step_total_tokens = [int(mask.numel()) for mask in qdp_masks]
    per_step_kept_tokens = [int(mask.sum().item()) for mask in qdp_masks]
    per_step_dropped_tokens = [total - kept for total, kept in zip(per_step_total_tokens, per_step_kept_tokens)]
    stats = QueryStreamStats(
        keep_rate=float(visual_keep.float().mean().item()),
        per_frame_keep_rates=keep_rates,
        per_step_total_tokens=per_step_total_tokens,
        per_step_kept_tokens=per_step_kept_tokens,
        per_step_dropped_tokens=per_step_dropped_tokens,
        relevance_scores=relevance_scores,
    )
    return QueryStreamVisualMasks(
        masks=qdp_masks,
        qwen_grid_hw=(qwen_h, qwen_w),
        stats=stats,
    )


def _build_messages(video_value, query: str):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_value,
                },
                {"type": "text", "text": query},
            ],
        }
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QueryStream CLI inference on a local video.")
    parser.add_argument("--video", required=True, help="Path to the local video file.")
    parser.add_argument("--query", required=True, help="User query for video understanding.")
    parser.add_argument("--checkpoint-path", default="wyccccc/TimeChatOnline-7B")
    parser.add_argument("--device", default="cpu", help="Main model device when --device-map none is used.")
    parser.add_argument(
        "--device-map",
        default="cpu",
        help="Model placement: cpu, auto, or none. Use cpu on Mac to avoid disk/meta offload issues.",
    )
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--flash-attn2", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=1016)
    parser.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default=DEFAULT_LOCAL_CLIP_PRETRAINED)
    parser.add_argument("--clip-device", default=None)
    parser.add_argument("--clip-batch-size", type=int, default=64, help="Batch size for CLIP scoring of Qwen token crops.")
    parser.add_argument(
        "--qdp-align-method",
        choices=("overlap", "nearest", "crop"),
        default="overlap",
        help=(
            "How to align CLIP query relevance to Qwen visual tokens: "
            "overlap=full-frame CLIP patch features with overlap pooling; "
            "nearest=full-frame CLIP patch mask with nearest resize; "
            "crop=score each Qwen token crop directly."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tau-temp", type=float, default=0.90)
    parser.add_argument("--output-json", default=None, help="Optional path to save response and QueryStream stats.")
    parser.add_argument("--verbose", action="store_true", help="Print input size diagnostics.")
    parser.add_argument(
        "--print-qdp-stats",
        action="store_true",
        help="Print before/after visual-token counts for every Qwen temporal step.",
    )
    parser.add_argument(
        "--print-token-stats",
        action="store_true",
        help="Print actual pre/post-pruning token counts inside the model prefill forward pass.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    local_video_path = args.video[7:] if args.video.startswith("file://") else args.video
    if not os.path.exists(local_video_path):
        raise FileNotFoundError(local_video_path)

    model_device_map = {"": "cpu"} if args.cpu_only else _normalize_device_map(args.device_map)
    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": model_device_map,
    }
    if model_device_map is None:
        model_kwargs.pop("device_map")
    if args.flash_attn2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"Loading video-language model from {args.checkpoint_path}...", file=sys.stderr)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.checkpoint_path, **model_kwargs).eval()
    if model_device_map is None:
        model = model.to(torch.device(args.device))
    processor = AutoProcessor.from_pretrained(args.checkpoint_path)

    clip_device_name = args.clip_device or ("cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda")
    clip_device = torch.device(clip_device_name)
    print(f"Loading OpenCLIP {args.clip_model} ({args.clip_pretrained}) on {clip_device}...", file=sys.stderr)
    clip_model, clip_preprocess, clip_tokenizer = _load_openclip(args.clip_model, args.clip_pretrained, clip_device)

    video_frames = load_video_frames(
        local_video_path,
        fps=args.fps,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    messages = _build_messages(video_frames, args.query)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=None,
        videos=[video_frames],
        padding=True,
        return_tensors="pt",
    )
    if args.verbose:
        print(f"sampled_frames={len(video_frames)}", file=sys.stderr)
        print(f"frame_size={video_frames[0].size}", file=sys.stderr)
        print(f"input_ids_shape={tuple(inputs.input_ids.shape)}", file=sys.stderr)
        if "pixel_values_videos" in inputs:
            print(f"pixel_values_videos_shape={tuple(inputs.pixel_values_videos.shape)}", file=sys.stderr)
        if "video_grid_thw" in inputs:
            print(f"video_grid_thw={inputs.video_grid_thw.tolist()}", file=sys.stderr)

    target_device = _first_real_device(model)
    inputs = inputs.to(target_device)

    visual_embedded, visual_embedded_type = model.embedding_visual_info(**inputs)
    if visual_embedded_type != "video":
        raise ValueError("QueryStream CLI expected a video input.")

    keep_mask, stats = build_querystream_keep_mask(
        video_tensor=video_frames,
        video_grid_thw=inputs["video_grid_thw"],
        input_ids=inputs["input_ids"],
        vision_start_token_id=model.config.vision_start_token_id,
        vision_end_token_id=model.config.vision_end_token_id,
        query=args.query,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer,
        clip_device=clip_device,
        alpha=args.alpha,
        tau_temp=args.tau_temp,
        clip_batch_size=args.clip_batch_size,
        qdp_align_method=args.qdp_align_method,
    )

    print(f"QueryStream visual keep rate: {stats.keep_rate:.4f}", file=sys.stderr)
    if args.print_qdp_stats:
        print_qdp_token_stats(stats, args.fps, label="QDP")

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
    generated_ids = model.generate(
        **inputs,
        visual_embedded=visual_embedded,
        visual_embedded_type=visual_embedded_type,
        keep_mask=keep_mask,
        drop_method="feature",
        drop_threshold=0.0,
        drop_absolute=True,
        print_token_stats=args.print_token_stats,
        **generation_kwargs,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(response)
    if args.output_json:
        output_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        payload = {
            "video": os.path.abspath(args.video),
            "query": args.query,
            "response": response,
            "querystream": {
                "keep_rate": stats.keep_rate,
                "per_frame_keep_rates": stats.per_frame_keep_rates,
                "per_step_total_tokens": stats.per_step_total_tokens,
                "per_step_kept_tokens": stats.per_step_kept_tokens,
                "per_step_dropped_tokens": stats.per_step_dropped_tokens,
                "relevance_scores": stats.relevance_scores,
                "alpha": args.alpha,
                "tau_temp": args.tau_temp,
            },
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
