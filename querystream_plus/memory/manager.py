"""Complementary token memory for QueryStream++."""

from __future__ import annotations

from dataclasses import InitVar, asdict, dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


MemoryPath = Literal["qim", "im"]
MemoryKind = Literal["prototype", "transition_before", "transition_after"]


@dataclass
class TokenMetadata:
    timestamp: float
    frame_id: int
    row: int
    col: int
    grid_h: int
    grid_w: int


@dataclass
class MemorySlot:
    slot_id: int
    path: MemoryPath
    key: torch.Tensor
    value: torch.Tensor
    timestamp: float
    frame_id: int
    row: int
    col: int
    grid_h: int
    grid_w: int
    source_turn: int
    episode_id: Optional[int] = None
    kind: MemoryKind = "prototype"
    novelty: float = 0.0
    router_score: float = 0.0
    support: int = 1
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    key_is_normalized: InitVar[bool] = False

    def __post_init__(self, key_is_normalized: bool) -> None:
        self.key = (
            self.key.detach().float().flatten().contiguous()
            if key_is_normalized
            else _normalize_key(self.key)
        )
        self.value = self.value.detach().contiguous()
        if self.time_start is None:
            self.time_start = self.timestamp
        if self.time_end is None:
            self.time_end = self.timestamp


@dataclass
class EpisodeRecord:
    episode_id: int
    key: torch.Tensor
    start_time: float
    end_time: float
    slot_ids: List[int]
    source_turn: int
    previous_episode_id: Optional[int] = None
    next_episode_id: Optional[int] = None

    def __post_init__(self) -> None:
        self.key = _normalize_key(self.key)


@dataclass
class RetrievalResult:
    qim_slots: List[MemorySlot] = field(default_factory=list)
    im_slots: List[MemorySlot] = field(default_factory=list)

    @property
    def slots(self) -> List[MemorySlot]:
        unique: Dict[int, MemorySlot] = {}
        for slot in self.qim_slots + self.im_slots:
            unique[slot.slot_id] = slot
        return sorted(unique.values(), key=_chronological_slot_key)


def _normalize_key(key: torch.Tensor) -> torch.Tensor:
    key = key.detach().float().flatten().contiguous()
    return F.normalize(key, dim=0, eps=1e-12)


def _chronological_slot_key(slot: MemorySlot) -> Tuple[float, int, int, int, int]:
    return (slot.timestamp, slot.frame_id, slot.row, slot.col, slot.slot_id)


def _stable_argmax_tensor(values: torch.Tensor, stable_ids: torch.Tensor) -> torch.Tensor:
    """Device-resident stable argmax with the same minimum-ID tie break."""
    best = values.max()
    sentinel = torch.iinfo(stable_ids.dtype).max
    candidate_ids = torch.where(values == best, stable_ids, sentinel)
    return torch.argmin(candidate_ids)


def weighted_farthest_first(
    keys: torch.Tensor,
    utility: torch.Tensor,
    budget: int,
    stable_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    """Deterministic weighted k-center selection using cosine distance."""
    if keys.ndim != 2:
        raise ValueError("keys must have shape [N, D].")
    count = keys.shape[0]
    if count == 0 or budget <= 0:
        return []
    if utility.numel() != count:
        raise ValueError("utility must contain one value per key.")
    stable_ids = list(range(count)) if stable_ids is None else list(stable_ids)
    if len(stable_ids) != count:
        raise ValueError("stable_ids must contain one value per key.")

    keys = F.normalize(keys.float(), dim=-1, eps=1e-12)
    utility = utility.to(device=keys.device, dtype=torch.float32).flatten().clamp_min(0.0)
    utility = torch.where(utility.max() == 0.0, torch.ones_like(utility), utility)
    stable_tensor = torch.tensor(stable_ids, dtype=torch.long, device=keys.device)

    target = min(int(budget), count)
    selected = torch.empty(target, dtype=torch.long, device=keys.device)
    selected[0] = _stable_argmax_tensor(utility, stable_tensor)
    min_distance = (1.0 - keys @ keys[selected[0]]).clamp_min(0.0)
    for position in range(1, target):
        score = utility * min_distance
        score[selected[:position]] = -1.0
        next_idx = _stable_argmax_tensor(score, stable_tensor)
        selected[position] = next_idx
        distance = (1.0 - keys @ keys[next_idx]).clamp_min(0.0)
        min_distance = torch.minimum(min_distance, distance)
    return selected.cpu().tolist()


class ComplementaryTokenMemory:
    """CTM: fixed-budget Query-Independent Memory and interaction consolidation."""

    def __init__(
        self,
        qim_capacity: int = 1024,
        im_capacity: int = 2048,
        active_buffer_capacity: int = 2048,
        im_episode_budget: int = 128,
        im_transition_ratio: float = 1.0 / 3.0,
        local_cell_size: int = 2,
    ) -> None:
        if qim_capacity < 0 or im_capacity < 0 or active_buffer_capacity <= 0:
            raise ValueError("Memory capacities must be non-negative and active buffer capacity must be positive.")
        if im_episode_budget <= 0:
            raise ValueError("im_episode_budget must be positive.")
        if not 0.0 <= im_transition_ratio < 1.0:
            raise ValueError("im_transition_ratio must be in [0, 1).")
        self.qim_capacity = int(qim_capacity)
        self.im_capacity = int(im_capacity)
        self.active_buffer_capacity = int(active_buffer_capacity)
        self.im_episode_budget = int(im_episode_budget)
        self.im_transition_ratio = float(im_transition_ratio)
        self.local_cell_size = int(local_cell_size)

        self.qim_slots: Dict[int, MemorySlot] = {}
        self.im_slots: Dict[int, MemorySlot] = {}
        self.episodes: Dict[int, EpisodeRecord] = {}
        self._active_by_turn: Dict[int, List[MemorySlot]] = {}
        self._next_slot_id = 0
        self._next_episode_id = 0
        self.device: Optional[torch.device] = None

    def _new_slot(
        self,
        path: MemoryPath,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: TokenMetadata,
        turn_id: int,
        novelty: float,
        router_score: float,
        kind: MemoryKind = "prototype",
        episode_id: Optional[int] = None,
    ) -> MemorySlot:
        slot = MemorySlot(
            slot_id=self._next_slot_id,
            path=path,
            key=key,
            value=value,
            timestamp=float(metadata.timestamp),
            frame_id=int(metadata.frame_id),
            row=int(metadata.row),
            col=int(metadata.col),
            grid_h=int(metadata.grid_h),
            grid_w=int(metadata.grid_w),
            source_turn=int(turn_id),
            episode_id=episode_id,
            kind=kind,
            novelty=float(novelty),
            router_score=float(router_score),
            key_is_normalized=True,
        )
        self._next_slot_id += 1
        return slot

    @staticmethod
    def _validate_observation(
        semantic_keys: torch.Tensor,
        qwen_values: torch.Tensor,
        novelty: torch.Tensor,
        router_scores: torch.Tensor,
        active_mask: torch.Tensor,
        token_metadata: Sequence[TokenMetadata],
    ) -> int:
        count = semantic_keys.shape[0]
        if semantic_keys.ndim != 2 or qwen_values.ndim != 2:
            raise ValueError("semantic_keys and qwen_values must have shape [N, D].")
        if qwen_values.shape[0] != count:
            raise ValueError("semantic_keys and qwen_values must have the same token count.")
        for name, tensor in (("novelty", novelty), ("router_scores", router_scores), ("active_mask", active_mask)):
            if tensor.numel() != count:
                raise ValueError(f"{name} must contain one value per token.")
        if len(token_metadata) != count:
            raise ValueError("token_metadata must contain one record per token.")
        return count

    def observe(
        self,
        semantic_keys: torch.Tensor,
        qwen_values: torch.Tensor,
        novelty: torch.Tensor,
        router_scores: torch.Tensor,
        active_mask: torch.Tensor,
        token_metadata: Sequence[TokenMetadata],
        turn_id: int,
    ) -> None:
        """Write unselected tokens to QIM and buffer selected tokens for IM."""
        count = self._validate_observation(
            semantic_keys, qwen_values, novelty, router_scores, active_mask, token_metadata
        )
        target_device = semantic_keys.device
        if self.device is None:
            self.device = target_device
        elif self.device != target_device:
            self._move_storage(target_device)
        semantic_keys = F.normalize(
            semantic_keys.detach().to(device=target_device, dtype=torch.float32),
            dim=-1,
            eps=1e-12,
        )
        qwen_values = qwen_values.detach().to(device=target_device)
        # Transfer vectors once before attaching metadata.
        novelty = novelty.detach().float().flatten().cpu()
        router_scores = router_scores.detach().float().flatten().cpu()
        active_mask = active_mask.detach().bool().flatten().cpu()

        qim_candidates: List[MemorySlot] = []
        active_candidates = self._active_by_turn.setdefault(int(turn_id), [])
        for idx in range(count):
            path: MemoryPath = "im" if bool(active_mask[idx]) else "qim"
            slot = self._new_slot(
                path=path,
                key=semantic_keys[idx],
                value=qwen_values[idx],
                metadata=token_metadata[idx],
                turn_id=turn_id,
                novelty=float(novelty[idx]),
                router_score=float(router_scores[idx]),
            )
            if path == "im":
                active_candidates.append(slot)
            else:
                qim_candidates.append(slot)

        if qim_candidates:
            self._update_qim(self._local_reduce(qim_candidates))
        if len(active_candidates) > self.active_buffer_capacity:
            selected = self._select_slots(
                active_candidates,
                utility=torch.tensor(
                    [slot.router_score for slot in active_candidates],
                    device=target_device,
                ),
                budget=self.active_buffer_capacity,
            )
            self._active_by_turn[int(turn_id)] = selected

    def _local_reduce(self, candidates: Sequence[MemorySlot]) -> List[MemorySlot]:
        if self.local_cell_size <= 1:
            return list(candidates)
        groups: Dict[Tuple[int, int, int], List[MemorySlot]] = {}
        for slot in candidates:
            key = (
                slot.frame_id,
                slot.row // self.local_cell_size,
                slot.col // self.local_cell_size,
            )
            groups.setdefault(key, []).append(slot)

        ordered_groups = [(key, groups[key]) for key in sorted(groups)]
        medoid_indices: Dict[Tuple[int, int, int], int] = {}
        groups_by_size: Dict[int, List[Tuple[Tuple[int, int, int], List[MemorySlot]]]] = {}
        for key, group in ordered_groups:
            if len(group) > 1:
                groups_by_size.setdefault(len(group), []).append((key, group))
        # Batch local-cell medoid computation.
        for size, sized_groups in groups_by_size.items():
            batched_keys = torch.stack(
                [torch.stack([slot.key for slot in group]) for _, group in sized_groups]
            )
            centroids = F.normalize(batched_keys.mean(dim=1), dim=-1, eps=1e-12)
            scores = (batched_keys * centroids[:, None, :]).sum(dim=-1)
            indices = torch.argmax(scores, dim=1).cpu().tolist()
            for (key, _), medoid_idx in zip(sized_groups, indices):
                medoid_indices[key] = medoid_idx

        reduced: List[MemorySlot] = []
        for key, group in ordered_groups:
            novelty_slot = max(group, key=lambda slot: (slot.novelty, -slot.slot_id))
            reduced.append(novelty_slot)
            if len(group) > 1:
                medoid = group[medoid_indices[key]]
                if medoid.slot_id != novelty_slot.slot_id:
                    reduced.append(medoid)
        return sorted(reduced, key=_chronological_slot_key)

    @staticmethod
    def _select_slots(slots: Sequence[MemorySlot], utility: torch.Tensor, budget: int) -> List[MemorySlot]:
        if not slots or budget <= 0:
            return []
        keys = torch.stack([slot.key for slot in slots])
        indices = weighted_farthest_first(
            keys,
            utility.to(device=keys.device),
            budget,
            stable_ids=[slot.slot_id for slot in slots],
        )
        return [slots[idx] for idx in indices]

    def _update_qim(self, candidates: Sequence[MemorySlot]) -> None:
        if self.qim_capacity == 0:
            self.qim_slots.clear()
            return
        pool = list(self.qim_slots.values()) + list(candidates)
        if not pool:
            return
        selected = self._select_slots(
            pool,
            utility=torch.tensor(
                [max(slot.novelty, 0.0) for slot in pool],
                device=pool[0].key.device,
            ),
            budget=self.qim_capacity,
        )
        keys = torch.stack([slot.key for slot in pool])
        selected_keys = torch.stack([slot.key for slot in selected])
        assignments = torch.argmax(keys @ selected_keys.T, dim=1).cpu()

        consolidated: Dict[int, MemorySlot] = {}
        for center_idx, center in enumerate(selected):
            members = [pool[i] for i in (assignments == center_idx).nonzero(as_tuple=True)[0].tolist()]
            # Drop empty centers created by duplicate keys.
            if not members:
                continue
            if len(members) == 1:
                prototype = members[0]
                aggregate_key = prototype.key
                aggregate_support = max(prototype.support, 1)
            else:
                weights = torch.tensor(
                    [max(member.support, 1) for member in members],
                    dtype=torch.float32,
                    device=members[0].key.device,
                )
                member_keys = torch.stack([member.key for member in members])
                aggregate_key = F.normalize(
                    (member_keys * weights[:, None]).sum(dim=0),
                    dim=0,
                    eps=1e-12,
                )
                medoid_idx = int(torch.argmax(member_keys @ aggregate_key).item())
                prototype = members[medoid_idx]
                aggregate_support = int(weights.sum().item())
            prototype.key = aggregate_key
            prototype.support = aggregate_support
            prototype.novelty = max(member.novelty for member in members)
            prototype.time_start = min(float(member.time_start) for member in members)
            prototype.time_end = max(float(member.time_end) for member in members)
            prototype.path = "qim"
            consolidated[prototype.slot_id] = prototype
        self.qim_slots = consolidated

    def consolidate_turn(self, query_embedding: torch.Tensor, turn_id: int) -> Optional[EpisodeRecord]:
        active = self._active_by_turn.pop(int(turn_id), [])
        if not active or self.im_capacity == 0:
            return None

        episode_id = self._next_episode_id
        self._next_episode_id += 1
        total_budget = min(self.im_episode_budget, self.im_capacity, len(active))
        transition_slot_budget = min(
            int(round(total_budget * self.im_transition_ratio)), total_budget - 1
        )
        transition_slot_budget -= transition_slot_budget % 2
        transition_boundary_budget = transition_slot_budget // 2
        core_budget = max(total_budget - transition_slot_budget, 1)

        core = self._select_slots(
            active,
            utility=torch.tensor(
                [max(slot.router_score, 0.0) for slot in active],
                device=active[0].key.device,
            ),
            budget=core_budget,
        )
        transitions = self._transition_slots(active, transition_boundary_budget)
        chosen: Dict[int, MemorySlot] = {slot.slot_id: slot for slot in core + transitions}
        if len(chosen) < total_budget:
            remaining = [slot for slot in active if slot.slot_id not in chosen]
            fill = self._select_slots(
                remaining,
                utility=torch.tensor(
                    [max(slot.router_score, slot.novelty, 0.0) for slot in remaining],
                    device=remaining[0].key.device,
                ),
                budget=total_budget - len(chosen),
            )
            chosen.update({slot.slot_id: slot for slot in fill})

        episode_slots = sorted(chosen.values(), key=_chronological_slot_key)
        for slot in episode_slots:
            slot.path = "im"
            slot.episode_id = episode_id
            self.im_slots[slot.slot_id] = slot

        visual_key = F.normalize(torch.stack([slot.key for slot in episode_slots]).mean(dim=0), dim=0, eps=1e-12)
        q_key = _normalize_key(query_embedding)
        episode_key = F.normalize(q_key + visual_key, dim=0, eps=1e-12)
        previous_id = max(self.episodes) if self.episodes else None
        episode = EpisodeRecord(
            episode_id=episode_id,
            key=episode_key,
            start_time=min(slot.timestamp for slot in episode_slots),
            end_time=max(slot.timestamp for slot in episode_slots),
            slot_ids=[slot.slot_id for slot in episode_slots],
            source_turn=int(turn_id),
            previous_episode_id=previous_id,
        )
        if previous_id is not None:
            self.episodes[previous_id].next_episode_id = episode_id
        self.episodes[episode_id] = episode
        self._enforce_im_capacity()
        return self.episodes.get(episode_id)

    @staticmethod
    def _transition_slots(active: Sequence[MemorySlot], boundary_budget: int) -> List[MemorySlot]:
        if boundary_budget <= 0:
            return []
        by_frame: Dict[int, List[MemorySlot]] = {}
        for slot in active:
            by_frame.setdefault(slot.frame_id, []).append(slot)
        frame_ids = sorted(by_frame)
        if len(frame_ids) < 2:
            return []
        changes = []
        for idx in range(1, len(frame_ids)):
            after = by_frame[frame_ids[idx]]
            change = sum(slot.novelty for slot in after) / max(len(after), 1)
            changes.append((change, frame_ids[idx - 1], frame_ids[idx]))
        changes.sort(key=lambda item: (-item[0], item[1], item[2]))

        selected: List[MemorySlot] = []
        for _, before_id, after_id in changes[:boundary_budget]:
            before = max(by_frame[before_id], key=lambda slot: (slot.novelty, slot.router_score, -slot.slot_id))
            after = max(by_frame[after_id], key=lambda slot: (slot.novelty, slot.router_score, -slot.slot_id))
            before.kind = "transition_before"
            after.kind = "transition_after"
            selected.extend([before, after])
        return selected

    def _enforce_im_capacity(self) -> None:
        if len(self.im_slots) <= self.im_capacity:
            return
        episode_list = sorted(self.episodes.values(), key=lambda episode: episode.episode_id)
        keys = torch.stack([episode.key for episode in episode_list])
        selected_indices = weighted_farthest_first(
            keys,
            torch.ones(len(episode_list), device=keys.device),
            len(episode_list),
            stable_ids=[episode.episode_id for episode in episode_list],
        )
        keep_episode_ids: List[int] = []
        used = 0
        for idx in selected_indices:
            episode = episode_list[idx]
            cost = len(episode.slot_ids)
            remaining = self.im_capacity - used
            if remaining <= 0:
                break
            if cost <= remaining:
                keep_episode_ids.append(episode.episode_id)
                used += cost
            elif not keep_episode_ids:
                slots = [self.im_slots[slot_id] for slot_id in episode.slot_ids]
                trimmed = self._select_slots(
                    slots,
                    torch.ones(len(slots), device=slots[0].key.device),
                    remaining,
                )
                episode.slot_ids = [slot.slot_id for slot in trimmed]
                keep_episode_ids.append(episode.episode_id)
                used += len(trimmed)

        keep_ids = set(keep_episode_ids)
        for episode_id in list(self.episodes):
            if episode_id not in keep_ids:
                episode = self.episodes.pop(episode_id)
                for slot_id in episode.slot_ids:
                    self.im_slots.pop(slot_id, None)
        retained_slot_ids = {slot_id for episode in self.episodes.values() for slot_id in episode.slot_ids}
        self.im_slots = {slot_id: slot for slot_id, slot in self.im_slots.items() if slot_id in retained_slot_ids}
        self._repair_episode_links()

    def _repair_episode_links(self) -> None:
        retained = sorted(self.episodes)
        for idx, episode_id in enumerate(retained):
            episode = self.episodes[episode_id]
            episode.previous_episode_id = retained[idx - 1] if idx > 0 else None
            episode.next_episode_id = retained[idx + 1] if idx + 1 < len(retained) else None

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        qim_budget: int,
        im_budget: int,
        exclude_turn: Optional[int] = None,
    ) -> RetrievalResult:
        query_embedding = _normalize_key(query_embedding)
        if self.device is not None:
            query_embedding = query_embedding.to(self.device)
        qim = [
            slot for slot in self.qim_slots.values()
            if exclude_turn is None or slot.source_turn != int(exclude_turn)
        ]
        if qim and qim_budget > 0:
            similarities = torch.stack([slot.key for slot in qim]) @ query_embedding
            utility = similarities.clamp_min(0.0)
            if float(utility.max().item()) == 0.0:
                utility = torch.tensor(
                    [max(slot.novelty, 0.0) for slot in qim],
                    device=similarities.device,
                )
            qim_selected = self._select_slots(qim, utility, qim_budget)
        else:
            qim_selected = []

        im_selected = self._retrieve_im(query_embedding, im_budget)
        return RetrievalResult(
            qim_slots=sorted(qim_selected, key=_chronological_slot_key),
            im_slots=sorted(im_selected, key=_chronological_slot_key),
        )

    def _retrieve_im(self, query_embedding: torch.Tensor, budget: int) -> List[MemorySlot]:
        if budget <= 0 or not self.episodes:
            return []
        episode_values = list(self.episodes.values())
        similarities = (torch.stack([episode.key for episode in episode_values]) @ query_embedding).cpu().tolist()
        ranked = [
            episode
            for _, episode in sorted(
                zip(similarities, episode_values),
                key=lambda item: (-item[0], item[1].episode_id),
            )
        ]
        selected_episode_ids: List[int] = []
        selected_slots: Dict[int, MemorySlot] = {}

        def add_episode(episode_id: Optional[int]) -> None:
            if episode_id is None or episode_id in selected_episode_ids or episode_id not in self.episodes:
                return
            episode = self.episodes[episode_id]
            remaining = budget - len(selected_slots)
            if remaining <= 0:
                return
            slots = [self.im_slots[slot_id] for slot_id in episode.slot_ids if slot_id in self.im_slots]
            if len(slots) > remaining:
                utility = (torch.stack([slot.key for slot in slots]) @ query_embedding).clamp_min(0.0)
                slots = self._select_slots(slots, utility, remaining)
            selected_episode_ids.append(episode_id)
            selected_slots.update({slot.slot_id: slot for slot in slots})

        for episode in ranked:
            if len(selected_slots) >= budget:
                break
            add_episode(episode.episode_id)
            if len(selected_slots) < budget:
                add_episode(episode.previous_episode_id)
        return sorted(selected_slots.values(), key=_chronological_slot_key)

    def clear_active_turn(self, turn_id: int) -> None:
        self._active_by_turn.pop(int(turn_id), None)

    def _move_storage(self, device: torch.device) -> None:
        device = torch.device(device)
        for slot in list(self.qim_slots.values()) + list(self.im_slots.values()):
            slot.key = slot.key.to(device)
            slot.value = slot.value.to(device)
        for slots in self._active_by_turn.values():
            for slot in slots:
                slot.key = slot.key.to(device)
                slot.value = slot.value.to(device)
        for episode in self.episodes.values():
            episode.key = episode.key.to(device)
        self.device = device

    def stats(self) -> Dict[str, int]:
        return {
            "qim_slots": len(self.qim_slots),
            "im_slots": len(self.im_slots),
            "episodes": len(self.episodes),
            "active_tokens": sum(len(slots) for slots in self._active_by_turn.values()),
        }

    @staticmethod
    def _slot_to_state(slot: MemorySlot) -> dict:
        state = asdict(slot)
        state["key"] = slot.key.cpu()
        state["value"] = slot.value.cpu()
        return state

    @staticmethod
    def _episode_to_state(episode: EpisodeRecord) -> dict:
        state = asdict(episode)
        state["key"] = episode.key.cpu()
        return state

    def state_dict(self) -> dict:
        return {
            "config": {
                "qim_capacity": self.qim_capacity,
                "im_capacity": self.im_capacity,
                "active_buffer_capacity": self.active_buffer_capacity,
                "im_episode_budget": self.im_episode_budget,
                "im_transition_ratio": self.im_transition_ratio,
                "local_cell_size": self.local_cell_size,
            },
            "qim_slots": [self._slot_to_state(slot) for slot in self.qim_slots.values()],
            "im_slots": [self._slot_to_state(slot) for slot in self.im_slots.values()],
            "episodes": [self._episode_to_state(episode) for episode in self.episodes.values()],
            "active_by_turn": {
                turn_id: [self._slot_to_state(slot) for slot in slots]
                for turn_id, slots in self._active_by_turn.items()
            },
            "next_slot_id": self._next_slot_id,
            "next_episode_id": self._next_episode_id,
        }

    def load_state_dict(self, state: dict) -> None:
        config = dict(state["config"])
        for name, value in config.items():
            setattr(self, name, value)
        self.qim_slots = {
            item["slot_id"]: MemorySlot(**item) for item in state.get("qim_slots", [])
        }
        self.im_slots = {
            item["slot_id"]: MemorySlot(**item) for item in state.get("im_slots", [])
        }
        self.episodes = {
            item["episode_id"]: EpisodeRecord(**item) for item in state.get("episodes", [])
        }
        self._active_by_turn = {
            int(turn_id): [MemorySlot(**item) for item in slots]
            for turn_id, slots in state.get("active_by_turn", {}).items()
        }
        self._next_slot_id = int(state.get("next_slot_id", 0))
        self._next_episode_id = int(state.get("next_episode_id", 0))
        all_slots = list(self.qim_slots.values()) + list(self.im_slots.values())
        self.device = all_slots[0].key.device if all_slots else None

    @classmethod
    def from_state_dict(cls, state: dict) -> "ComplementaryTokenMemory":
        manager = cls(**state["config"])
        manager.load_state_dict(state)
        return manager
