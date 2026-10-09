"""Query-aligned visual evidence used by QueryStream++ routing and memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from querystream_plus.memory.manager import TokenMetadata
from querystream_plus.inference.vision import (
    _pool_siglip2_features_to_qwen_grid,
    _prepare_siglip2_batch,
    _siglip2_patch_features,
    _siglip2_text_feature,
    _video_input_to_pil_frames,
)
from querystream_plus.modeling.router import normalize_prior_scores


@dataclass
class VisualEvidence:
    semantic_keys: torch.Tensor
    query_embedding: torch.Tensor
    relevance: torch.Tensor
    novelty: torch.Tensor
    prior_score: torch.Tensor
    prior_features: torch.Tensor
    metadata: List[TokenMetadata]
    qwen_grid_hw: Tuple[int, int]
    temporal_state: torch.Tensor

def _local_window_novelty(current: torch.Tensor, history: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Return 1 - best local cosine match for a [H, W, D] feature grid."""
    current = F.normalize(current, dim=-1, eps=1e-12)
    history = F.normalize(history, dim=-1, eps=1e-12)
    height, width, _ = current.shape
    best = current.new_full((height, width), -1.0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            cur_y0, cur_y1 = max(0, -dy), min(height, height - dy)
            cur_x0, cur_x1 = max(0, -dx), min(width, width - dx)
            hist_y0, hist_y1 = cur_y0 + dy, cur_y1 + dy
            hist_x0, hist_x1 = cur_x0 + dx, cur_x1 + dx
            similarity = (
                current[cur_y0:cur_y1, cur_x0:cur_x1]
                * history[hist_y0:hist_y1, hist_x0:hist_x1]
            ).sum(dim=-1)
            best[cur_y0:cur_y1, cur_x0:cur_x1] = torch.maximum(
                best[cur_y0:cur_y1, cur_x0:cur_x1], similarity
            )
    return (1.0 - best).clamp_min(0.0)


@torch.no_grad()
def encode_aligned_semantic_tokens(
    video_frames,
    video_grid_thw: torch.Tensor,
    semantic_encoder,
    semantic_preprocess,
    semantic_device: torch.device,
):
    """Run only the frozen semantic image tower and align it to Qwen tokens."""
    if video_grid_thw is None or len(video_grid_thw) != 1:
        raise ValueError("Visual evidence encoding expects exactly one video chunk.")
    grid_t, grid_h, grid_w = [int(value) for value in video_grid_thw[0].tolist()]
    qwen_h, qwen_w = grid_h // 2, grid_w // 2
    frames = _video_input_to_pil_frames(video_frames)
    semantic_images = _prepare_siglip2_batch(frames, semantic_preprocess, semantic_device)
    patch_features = F.normalize(_siglip2_patch_features(semantic_encoder, semantic_images), dim=-1, eps=1e-12)
    aligned = torch.stack(
        [_pool_siglip2_features_to_qwen_grid(frame, (qwen_h, qwen_w)) for frame in patch_features],
        dim=0,
    )
    aligned = F.normalize(aligned, dim=-1, eps=1e-12)
    frame_indices = [min(index * 2 + 1, aligned.shape[0] - 1) for index in range(grid_t)]
    return aligned[frame_indices].detach(), (qwen_h, qwen_w)


@torch.no_grad()
def build_visual_evidence(
    aligned_semantic_tokens: torch.Tensor,
    video_grid_thw: torch.Tensor,
    query: str,
    semantic_encoder,
    semantic_processor,
    semantic_device: torch.device,
    temporal_alpha: float = 0.1,
    novelty_weight: float = 1.0,
    fps: float = 1.0,
    global_temporal_offset: int = 0,
    initial_dsh: Optional[torch.Tensor] = None,
    output_device: Optional[torch.device] = None,
) -> VisualEvidence:
    """Add query/stream-dependent signals to cached semantic image keys."""
    if video_grid_thw is None or len(video_grid_thw) != 1:
        raise ValueError("Visual evidence encoding expects exactly one video chunk.")
    grid_t, grid_h, grid_w = [int(value) for value in video_grid_thw[0].tolist()]
    qwen_h, qwen_w = grid_h // 2, grid_w // 2
    # Match cached keys to the query dtype below.
    aligned = aligned_semantic_tokens.to(semantic_device)
    # Accept flattened or explicit spatial grids.
    if aligned.ndim == 3 and aligned.shape[:2] == (grid_t, qwen_h * qwen_w):
        aligned = aligned.reshape(grid_t, qwen_h, qwen_w, aligned.shape[-1])
    if aligned.shape[:3] != (grid_t, qwen_h, qwen_w):
        raise ValueError(
            f"cached SigLIP2/Qwen grid mismatch: got {tuple(aligned.shape[:3])}, "
            f"expected {(grid_t, qwen_h, qwen_w)}"
        )
    query_embedding = _siglip2_text_feature(
        semantic_encoder, semantic_processor, query, semantic_device
    )
    aligned = F.normalize(aligned.to(dtype=query_embedding.dtype), dim=-1, eps=1e-12)

    keys: List[torch.Tensor] = []
    relevance_values: List[torch.Tensor] = []
    novelty_values: List[torch.Tensor] = []
    metadata: List[TokenMetadata] = []
    dsh = None if initial_dsh is None else initial_dsh.to(device=aligned.device, dtype=aligned.dtype)
    if dsh is not None and dsh.shape != (qwen_h, qwen_w, aligned.shape[-1]):
        raise ValueError("initial_dsh shape does not match the current aligned Qwen grid.")
    for local_t in range(grid_t):
        grid = aligned[local_t].view(qwen_h, qwen_w, -1)
        relevance = (grid @ query_embedding).flatten()
        if dsh is None:
            novelty = torch.ones(qwen_h, qwen_w, device=grid.device, dtype=grid.dtype)
            dsh = grid.clone()
        else:
            novelty = _local_window_novelty(grid, dsh)
            dsh = temporal_alpha * grid + (1.0 - temporal_alpha) * dsh

        keys.append(grid.flatten(0, 1))
        relevance_values.append(relevance)
        novelty_values.append(novelty.flatten())
        global_t = global_temporal_offset + local_t
        timestamp = (global_t + 1) * 2.0 / fps
        for row in range(qwen_h):
            for col in range(qwen_w):
                metadata.append(
                    TokenMetadata(
                        timestamp=timestamp,
                        frame_id=global_t,
                        row=row,
                        col=col,
                        grid_h=qwen_h,
                        grid_w=qwen_w,
                    )
                )

    semantic_keys = torch.cat(keys, dim=0)
    relevance = torch.cat(relevance_values, dim=0)
    novelty = torch.cat(novelty_values, dim=0)
    raw_prior = relevance + novelty_weight * novelty
    prior_score = normalize_prior_scores(raw_prior.unsqueeze(0)).squeeze(0)
    prior_features = torch.stack([prior_score, relevance, novelty], dim=-1)
    output_device = torch.device("cpu") if output_device is None else torch.device(output_device)
    return VisualEvidence(
        semantic_keys=semantic_keys.detach().to(output_device),
        query_embedding=query_embedding.detach().to(output_device),
        relevance=relevance.detach().to(output_device),
        novelty=novelty.detach().to(output_device),
        prior_score=prior_score.detach().to(output_device),
        prior_features=prior_features.detach().to(output_device),
        metadata=metadata,
        qwen_grid_hw=(qwen_h, qwen_w),
        temporal_state=dsh.detach().to(output_device),
    )


@torch.no_grad()
def encode_visual_evidence(
    video_frames,
    video_grid_thw: torch.Tensor,
    query: str,
    semantic_encoder,
    semantic_preprocess,
    semantic_processor,
    semantic_device: torch.device,
    **kwargs,
) -> VisualEvidence:
    """Encode frames and build query-aware visual evidence."""
    aligned, _ = encode_aligned_semantic_tokens(
        video_frames=video_frames,
        video_grid_thw=video_grid_thw,
        semantic_encoder=semantic_encoder,
        semantic_preprocess=semantic_preprocess,
        semantic_device=semantic_device,
    )
    return build_visual_evidence(
        aligned_semantic_tokens=aligned,
        video_grid_thw=video_grid_thw,
        query=query,
        semantic_encoder=semantic_encoder,
        semantic_processor=semantic_processor,
        semantic_device=semantic_device,
        **kwargs,
    )
