"""Qwen2.5-VL prefill packing support for QueryStream++ memory inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from querystream_plus.modeling.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from querystream_plus.memory.manager import MemorySlot, TokenMetadata


@dataclass
class PackedVisualContext:
    input_ids: torch.LongTensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.LongTensor
    position_ids: torch.LongTensor
    visual_sources: List[str]
    visual_metadata: List[TokenMetadata]
    num_memory_tokens: int
    num_current_tokens: int

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.shape[1])


class QueryStreamPlusForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Backbone wrapper with memory-aware M-RoPE."""

    def set_memory_prefill_positions(self, position_ids: torch.Tensor, sequence_length: int) -> None:
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError("Memory position_ids must have shape [3, batch, sequence].")
        max_position = position_ids.amax(dim=(0, 2)).view(-1, 1)
        self._memory_rope_delta = max_position + 1 - int(sequence_length)

    def clear_memory_prefill_positions(self) -> None:
        self._memory_rope_delta = None

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        delta = getattr(self, "_memory_rope_delta", None)
        if (
            delta is not None
            and cache_position is not None
            and cache_position.numel() > 0
            and int(cache_position[0].item()) > 0
        ):
            batch_size = input_ids.shape[0]
            delta = delta.to(input_ids.device)
            if delta.shape[0] == 1 and batch_size > 1:
                delta = delta.expand(batch_size, -1)
            next_positions = cache_position.view(1, 1, -1) + delta.view(1, batch_size, 1)
            position_ids = next_positions.expand(3, -1, -1)
        return super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )


def _validate_single_video_span(input_ids: torch.Tensor, video_token_id: int) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("QueryStream++ visual context packing supports batch size 1.")
    visual_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
    if visual_positions.numel() == 0:
        raise ValueError("The prompt must contain one non-empty video span.")
    expected = torch.arange(
        int(visual_positions[0]),
        int(visual_positions[-1]) + 1,
        device=visual_positions.device,
    )
    if not torch.equal(visual_positions, expected):
        raise ValueError("QueryStream++ expects one contiguous video-token span.")
    return visual_positions


def _remapped_visual_positions(
    metadata: Sequence[TokenMetadata],
    visual_base: int,
    device: torch.device,
) -> torch.LongTensor:
    if not metadata:
        return torch.empty(3, 0, dtype=torch.long, device=device)
    distinct_times = sorted({float(item.timestamp) for item in metadata})
    time_rank = {timestamp: rank for rank, timestamp in enumerate(distinct_times)}
    temporal = torch.tensor([time_rank[float(item.timestamp)] for item in metadata], device=device)
    height = torch.tensor([int(item.row) for item in metadata], device=device)
    width = torch.tensor([int(item.col) for item in metadata], device=device)
    return torch.stack([temporal, height, width], dim=0).long() + int(visual_base)


def pack_visual_context(
    model: QueryStreamPlusForConditionalGeneration,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    current_visual_values: torch.Tensor,
    current_keep_mask: torch.BoolTensor,
    current_metadata: Sequence[TokenMetadata],
    recalled_slots: Sequence[MemorySlot],
) -> PackedVisualContext:
    """Replace the prompt video span by recalled plus selected current tokens."""
    visual_positions = _validate_single_video_span(input_ids, model.config.video_token_id)
    num_current = int(visual_positions.numel())
    if current_visual_values.ndim != 2 or current_visual_values.shape[0] != num_current:
        raise ValueError("current_visual_values must align one-to-one with prompt video tokens.")
    current_keep_mask = current_keep_mask.detach().bool().flatten()
    if current_keep_mask.numel() != num_current:
        raise ValueError("current_keep_mask must contain one decision per current video token.")
    if len(current_metadata) != num_current:
        raise ValueError("current_metadata must contain one record per current video token.")

    device = input_ids.device
    dtype = current_visual_values.dtype
    prefix_end = int(visual_positions[0])
    suffix_start = int(visual_positions[-1]) + 1
    prefix_ids = input_ids[:, :prefix_end]
    suffix_ids = input_ids[:, suffix_start:]

    recalled = sorted(
        recalled_slots,
        key=lambda slot: (slot.timestamp, slot.frame_id, slot.row, slot.col, slot.slot_id),
    )
    packed_items: List[Tuple[float, int, int, int, str, torch.Tensor, TokenMetadata]] = []
    for slot in recalled:
        metadata = TokenMetadata(
            timestamp=slot.timestamp,
            frame_id=slot.frame_id,
            row=slot.row,
            col=slot.col,
            grid_h=slot.grid_h,
            grid_w=slot.grid_w,
        )
        packed_items.append(
            (slot.timestamp, slot.frame_id, slot.row, slot.col, slot.path, slot.value, metadata)
        )
    for index in current_keep_mask.nonzero(as_tuple=True)[0].tolist():
        metadata = current_metadata[index]
        packed_items.append(
            (
                metadata.timestamp,
                metadata.frame_id,
                metadata.row,
                metadata.col,
                "current",
                current_visual_values[index],
                metadata,
            )
        )
    packed_items.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    if not packed_items:
        raise ValueError("At least one recalled or current visual token is required.")

    visual_values = torch.stack([item[5].to(device=device, dtype=dtype) for item in packed_items])
    visual_metadata = [item[6] for item in packed_items]
    visual_sources = [item[4] for item in packed_items]
    visual_ids = input_ids.new_full((1, len(packed_items)), model.config.video_token_id)
    packed_ids = torch.cat([prefix_ids, visual_ids, suffix_ids], dim=1)

    text_embeds = model.model.embed_tokens(packed_ids)
    packed_embeds = text_embeds.clone()
    new_visual_start = prefix_ids.shape[1]
    packed_embeds[:, new_visual_start : new_visual_start + len(packed_items)] = visual_values.unsqueeze(0)
    packed_attention = torch.ones_like(packed_ids, dtype=attention_mask.dtype, device=device)

    prefix_positions = torch.arange(prefix_ids.shape[1], device=device).view(1, -1).expand(3, -1)
    visual_positions_3d = _remapped_visual_positions(
        visual_metadata,
        visual_base=prefix_ids.shape[1],
        device=device,
    )
    suffix_base = int(visual_positions_3d.max().item()) + 1
    suffix_positions = (
        torch.arange(suffix_ids.shape[1], device=device).view(1, -1).expand(3, -1) + suffix_base
    )
    packed_position_ids = torch.cat(
        [prefix_positions, visual_positions_3d, suffix_positions], dim=1
    ).unsqueeze(1)

    expected_length = packed_ids.shape[1]
    if packed_embeds.shape[1] != expected_length or packed_attention.shape[1] != expected_length:
        raise AssertionError("Packed ids, embeddings, and attention mask have different lengths.")
    if packed_position_ids.shape[-1] != expected_length:
        raise AssertionError("Packed M-RoPE position ids do not match the packed sequence length.")

    return PackedVisualContext(
        input_ids=packed_ids,
        inputs_embeds=packed_embeds,
        attention_mask=packed_attention,
        position_ids=packed_position_ids,
        visual_sources=visual_sources,
        visual_metadata=visual_metadata,
        num_memory_tokens=len(recalled),
        num_current_tokens=int(current_keep_mask.sum().item()),
    )
