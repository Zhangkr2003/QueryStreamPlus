import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_prior_scores(
    raw_scores: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Rank-normalize QueryStream prior scores to (0, 1)."""
    if raw_scores.dim() == 1:
        raw_scores = raw_scores.unsqueeze(0)

    order = raw_scores.argsort(dim=-1)
    ranks = torch.zeros_like(raw_scores, dtype=torch.float32)
    rank_values = torch.linspace(eps, 1.0 - eps, raw_scores.shape[-1], device=raw_scores.device)
    rank_values = rank_values.to(raw_scores.dtype)
    ranks.scatter_(dim=-1, index=order, src=rank_values.expand_as(raw_scores))
    return ranks.clamp(eps, 1.0 - eps)


def rescale_score_for_attention(score: torch.Tensor, min_score: float = math.exp(-2.0)) -> torch.Tensor:
    return min_score + score.clamp(0.0, 1.0) * (1.0 - min_score)


def build_router_attention_bias(
    input_ids: torch.LongTensor,
    video_token_id: int,
    visual_scores: torch.Tensor,
    min_score: float = math.exp(-2.0),
) -> torch.Tensor:
    """Build a [B, 1, 1, L] key-side additive attention bias for visual tokens."""
    if visual_scores.dim() == 1:
        visual_scores = visual_scores.unsqueeze(0)
    batch_size, seq_len = input_ids.shape
    bias = torch.zeros(batch_size, 1, 1, seq_len, dtype=visual_scores.dtype, device=visual_scores.device)
    visual_bias = torch.log(rescale_score_for_attention(visual_scores, min_score=min_score))
    for batch_idx in range(batch_size):
        positions = (input_ids[batch_idx] == video_token_id).nonzero(as_tuple=True)[0]
        if positions.numel() != visual_scores.shape[-1]:
            raise ValueError(
                f"visual score count ({visual_scores.shape[-1]}) does not match video token count ({positions.numel()})."
            )
        bias[batch_idx, :, :, positions] = visual_bias[batch_idx].view(1, 1, -1)
    return bias


def build_budgeted_keep_mask(
    input_ids: torch.LongTensor,
    video_token_id: int,
    visual_scores: torch.Tensor,
    keep_rate: float,
    always_keep_first_frame_tokens: int = 0,
) -> torch.BoolTensor:
    """Create a full-sequence keep mask from visual scores. The top-k decision is non-differentiable."""
    if visual_scores.dim() == 1:
        visual_scores = visual_scores.unsqueeze(0)
    if not 0.0 < keep_rate <= 1.0:
        raise ValueError("keep_rate must be in (0, 1].")

    full_keep = torch.ones_like(input_ids, dtype=torch.bool)
    num_visual = visual_scores.shape[-1]
    keep_n = max(1, int(round(num_visual * keep_rate)))
    keep_n = min(keep_n, num_visual)

    for batch_idx in range(input_ids.shape[0]):
        visual_positions = (input_ids[batch_idx] == video_token_id).nonzero(as_tuple=True)[0]
        if visual_positions.numel() != num_visual:
            raise ValueError("All samples in a router batch must have the same number of video tokens for now.")
        scores = visual_scores[batch_idx].detach()
        keep_visual = torch.zeros(num_visual, dtype=torch.bool, device=input_ids.device)
        if always_keep_first_frame_tokens > 0:
            first = min(always_keep_first_frame_tokens, num_visual)
            keep_visual[:first] = True
            remaining = max(keep_n - first, 0)
            if remaining > 0:
                candidate_scores = scores[first:]
                topk = torch.topk(candidate_scores, k=min(remaining, candidate_scores.numel()), largest=True).indices
                keep_visual[first + topk] = True
        else:
            topk = torch.topk(scores, k=keep_n, largest=True).indices
            keep_visual[topk] = True
        full_keep[batch_idx, visual_positions] = keep_visual
    return full_keep


def select_routed_tokens(
    visual_scores: torch.Tensor,
    active_budget: int,
) -> torch.BoolTensor:
    """Select the highest-scoring router tokens."""
    scores = visual_scores.detach().float().flatten()
    if active_budget <= 0:
        raise ValueError("active_budget must be positive.")

    keep = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    count = min(int(active_budget), scores.numel())
    # Preserve earlier tokens when scores tie.
    ranked = torch.argsort(scores, descending=True, stable=True)
    keep[ranked[:count]] = True
    return keep


@dataclass(frozen=True)
class RouterConfig:
    """Serializable architecture contract shared by training and inference."""

    hidden_size: int = 3584
    semantic_size: int = 1024
    prior_feature_size: int = 3
    router_dim: int = 768
    num_heads: int = 12
    ffn_ratio: float = 4.0
    spatial_window: int = 4
    temporal_window: int = 8
    dropout: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Optional[Dict], **fallback) -> "RouterConfig":
        payload = dict(fallback)
        if values:
            payload.update({key: value for key, value in values.items() if key in cls.__dataclass_fields__})
        return cls(**payload)


class _AttentionFFNBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        ffn_dim = int(round(dim * ffn_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = self.norm1(x)
        attended = self.attn(
            normed,
            normed,
            normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x + attended
        return x + self.ffn(self.norm2(x))


class _CrossAttentionFFNBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_ratio: float, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        ffn_dim = int(round(dim * ffn_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attended = self.attn(
            self.query_norm(visual),
            self.context_norm(text),
            self.context_norm(text),
            key_padding_mask=text_padding_mask,
            need_weights=False,
        )[0]
        visual = visual + attended
        return visual + self.ffn(self.norm2(visual))


class QueryAwareSpatiotemporalRouter(nn.Module):
    """Query-aware spatiotemporal residual router."""

    def __init__(self, config: RouterConfig):
        super().__init__()
        if config.router_dim % config.num_heads:
            raise ValueError("router_dim must be divisible by num_heads.")
        if config.spatial_window <= 0 or config.temporal_window <= 0:
            raise ValueError("Router window sizes must be positive.")
        self.config = config
        dim = config.router_dim
        self.visual_proj = nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, dim))
        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(config.semantic_size), nn.Linear(config.semantic_size, dim)
        )
        self.semantic_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.text_proj = nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, dim))
        self.cross_block = _CrossAttentionFFNBlock(
            dim, config.num_heads, config.ffn_ratio, config.dropout
        )
        self.spatial_block = _AttentionFFNBlock(
            dim, config.num_heads, config.ffn_ratio, config.dropout
        )
        self.temporal_block = _AttentionFFNBlock(
            dim, config.num_heads, config.ffn_ratio, config.dropout
        )
        self.prior_norm = nn.LayerNorm(config.prior_feature_size)
        self.score_head = nn.Sequential(
            nn.LayerNorm(dim + config.prior_feature_size),
            nn.Linear(dim + config.prior_feature_size, dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    @staticmethod
    def _pad_grid(x: torch.Tensor, pad_t: int, pad_h: int, pad_w: int) -> torch.Tensor:
        # F.pad works from the last dimension; keep the channel dimension intact.
        return F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))

    def _spatial_windows(self, x: torch.Tensor, grid_shape: Tuple[int, int, int]) -> torch.Tensor:
        batch, _, dim = x.shape
        time, height, width = grid_shape
        window = self.config.spatial_window
        pad_h = (-height) % window
        pad_w = (-width) % window
        grid = x.view(batch, time, height, width, dim)
        grid = self._pad_grid(grid, 0, pad_h, pad_w)
        padded_h, padded_w = height + pad_h, width + pad_w
        windows = (
            grid.view(batch, time, padded_h // window, window, padded_w // window, window, dim)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(-1, window * window, dim)
        )
        valid = None
        if pad_h or pad_w:
            valid = x.new_ones((batch, time, height, width), dtype=torch.bool)
            valid = F.pad(valid, (0, pad_w, 0, pad_h), value=False)
            valid = (
                valid.view(batch, time, padded_h // window, window, padded_w // window, window)
                .permute(0, 1, 2, 4, 3, 5)
                .reshape(-1, window * window)
            )
            windows = self.spatial_block(windows, key_padding_mask=~valid)
        else:
            windows = self.spatial_block(windows)
        if valid is not None:
            windows = windows.masked_fill(~valid.unsqueeze(-1), 0.0)
        grid = (
            windows.view(batch, time, padded_h // window, padded_w // window, window, window, dim)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(batch, time, padded_h, padded_w, dim)
        )
        return grid[:, :, :height, :width].reshape(batch, time * height * width, dim)

    def _temporal_windows(self, x: torch.Tensor, grid_shape: Tuple[int, int, int]) -> torch.Tensor:
        batch, _, dim = x.shape
        time, height, width = grid_shape
        window = self.config.temporal_window
        pad_t = (-time) % window
        grid = x.view(batch, time, height, width, dim)
        grid = self._pad_grid(grid, pad_t, 0, 0)
        padded_t = time + pad_t
        windows = (
            grid.view(batch, padded_t // window, window, height, width, dim)
            .permute(0, 3, 4, 1, 2, 5)
            .reshape(-1, window, dim)
        )
        valid = None
        if pad_t:
            valid = x.new_ones((batch, time, height, width), dtype=torch.bool)
            valid = F.pad(valid, (0, 0, 0, 0, 0, pad_t), value=False)
            valid = (
                valid.view(batch, padded_t // window, window, height, width)
                .permute(0, 3, 4, 1, 2)
                .reshape(-1, window)
            )
            windows = self.temporal_block(windows, key_padding_mask=~valid)
        else:
            windows = self.temporal_block(windows)
        if valid is not None:
            windows = windows.masked_fill(~valid.unsqueeze(-1), 0.0)
        grid = (
            windows.view(batch, height, width, padded_t // window, window, dim)
            .permute(0, 3, 4, 1, 2, 5)
            .reshape(batch, padded_t, height, width, dim)
        )
        return grid[:, :time].reshape(batch, time * height * width, dim)

    def forward(
        self,
        visual_hidden: torch.Tensor,
        text_context: torch.Tensor,
        prior_score: torch.Tensor,
        prior_features: Optional[torch.Tensor] = None,
        residual_scale: float = 1.0,
        semantic_hidden: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if visual_hidden.dim() != 3:
            raise ValueError("visual_hidden must have shape [batch, tokens, hidden_size].")
        if text_context.dim() == 2:
            text_context = text_context.unsqueeze(1)
        if text_context.dim() != 3:
            raise ValueError("text_context must contain pooled or token-level text features.")
        if grid_shape is None or math.prod(grid_shape) != visual_hidden.shape[1]:
            raise ValueError("grid_shape=(time, height, width) must match the visual token count.")
        if prior_score.dim() == 1:
            prior_score = prior_score.unsqueeze(0)
        prior_score = prior_score.to(visual_hidden.device, visual_hidden.dtype).clamp(1e-4, 1.0 - 1e-4)
        if prior_features is None:
            prior_features = torch.stack(
                [prior_score, torch.zeros_like(prior_score), torch.zeros_like(prior_score)], dim=-1
            )
        prior_features = prior_features.to(visual_hidden.device, visual_hidden.dtype)

        visual = self.visual_proj(visual_hidden)
        if semantic_hidden is not None:
            semantic_hidden = semantic_hidden.to(visual_hidden.device, visual_hidden.dtype)
            if semantic_hidden.shape[:2] != visual_hidden.shape[:2]:
                raise ValueError("semantic_hidden must align one-to-one with visual_hidden.")
            semantic = self.semantic_proj(semantic_hidden)
            gate = self.semantic_gate(torch.cat([visual, semantic], dim=-1))
            visual = gate * visual + (1.0 - gate) * semantic

        text = self.text_proj(text_context.to(visual_hidden.device, visual_hidden.dtype))
        padding_mask = None
        if text_attention_mask is not None:
            if text_attention_mask.shape != text.shape[:2]:
                raise ValueError("text_attention_mask must have shape [batch, text_tokens].")
            padding_mask = ~text_attention_mask.to(device=text.device, dtype=torch.bool)
        visual = self.cross_block(visual, text, padding_mask)
        visual = self._spatial_windows(visual, grid_shape)
        visual = self._temporal_windows(visual, grid_shape)
        score_input = torch.cat([visual, self.prior_norm(prior_features)], dim=-1)
        delta = self.score_head(score_input).squeeze(-1)
        refined_score = torch.sigmoid(torch.logit(prior_score) + residual_scale * delta)
        return refined_score, delta


def build_router(config: RouterConfig) -> nn.Module:
    return QueryAwareSpatiotemporalRouter(config)
