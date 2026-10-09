import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AnswerEvent:
    sample_id: str
    event_id: str
    video: str
    visible_video_start: float
    visible_video_end: float
    context_until: float
    answer_timestamp: Optional[float]
    dialogue_context: List[Dict[str, Any]]
    target: Dict[str, Any]
    training_mode: str = "causal_prefix"
    source_dataset: Optional[str] = None
    source_split: Optional[str] = None
    turn_index: int = 0
    frames: Optional[List[str]] = None
    frame_timestamps: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "event_id": self.event_id,
            "video": self.video,
            "visible_video_start": self.visible_video_start,
            "visible_video_end": self.visible_video_end,
            "context_until": self.context_until,
            "answer_timestamp": self.answer_timestamp,
            "dialogue_context": copy.deepcopy(self.dialogue_context),
            "target": copy.deepcopy(self.target),
            "training_mode": self.training_mode,
            "source_dataset": self.source_dataset,
            "source_split": self.source_split,
            "turn_index": self.turn_index,
            "frames": None if self.frames is None else list(self.frames),
            "frame_timestamps": None
            if self.frame_timestamps is None
            else list(self.frame_timestamps),
        }


def _timeline_time(node: Dict[str, Any], fallback: float) -> float:
    for key in ("time", "timestamp", "end", "answer_timestamp", "context_until"):
        if key in node and node[key] is not None:
            return float(node[key])
    return fallback


def _append_dialogue(history: List[Dict[str, Any]], node: Dict[str, Any], source_id: str) -> None:
    history.append(
        {
            "role": node.get("role", "user" if node.get("type") == "question" else "assistant"),
            "text": node.get("text", ""),
            "source_id": source_id,
        }
    )


def _answer_texts(node: Dict[str, Any]) -> List[str]:
    if "answers" in node:
        answers = node["answers"]
        if not isinstance(answers, list) or not answers:
            raise ValueError("multi_reference answer nodes require a non-empty `answers` list.")
        return [str(answer) for answer in answers]
    if "text" not in node:
        raise ValueError("answer nodes require either `text` or `answers`.")
    return [str(node["text"])]


def timeline_to_events(sample: Dict[str, Any]) -> List[AnswerEvent]:
    """Expand a mixed video/dialogue timeline into independently trainable answer events."""
    sample_id = str(sample.get("id", sample.get("sample_id", "")))
    if not sample_id:
        raise ValueError("Each sample requires `id` or `sample_id`.")
    video = sample.get("video")
    if video is None:
        raise ValueError(f"Sample {sample_id} requires `video`.")

    visible_video_start = float(sample.get("visible_video_start", 0.0))
    visible_video_end = float(sample.get("visible_video_end", 0.0))
    context_until = visible_video_end
    dialogue_history: List[Dict[str, Any]] = []
    events: List[AnswerEvent] = []
    training_mode = str(sample.get("training_mode", "causal_prefix"))
    if training_mode not in {"causal_prefix", "memory"}:
        raise ValueError(
            f"Sample {sample_id} has unsupported training_mode {training_mode!r}; "
            "expected 'causal_prefix' or 'memory'."
        )
    source_dataset = sample.get("source_dataset")
    source_split = sample.get("source_split")
    sample_frames = sample.get("frames")
    sample_frame_timestamps = sample.get("frame_timestamps")
    if sample_frames is not None:
        if not isinstance(sample_frames, list) or not sample_frames:
            raise ValueError(f"Sample {sample_id} frames must be a non-empty list when provided.")
        sample_frames = [str(path) for path in sample_frames]
        if sample_frame_timestamps is not None:
            sample_frame_timestamps = [float(value) for value in sample_frame_timestamps]
            if len(sample_frame_timestamps) != len(sample_frames):
                raise ValueError(f"Sample {sample_id} frame_timestamps must align with frames.")
    answer_turn_index = 0

    for idx, node in enumerate(sample.get("timeline", [])):
        node_type = node.get("type")
        source_id = str(node.get("id", node.get("qid", node.get("answer_id", f"node_{idx}"))))

        if node_type == "clip":
            if "start" in node and node["start"] is not None:
                visible_video_start = min(visible_video_start, float(node["start"]))
            if "end" in node and node["end"] is not None:
                visible_video_end = max(visible_video_end, float(node["end"]))
                context_until = visible_video_end
            else:
                context_until = _timeline_time(node, context_until)
            continue

        if node_type == "question":
            context_until = _timeline_time(node, context_until)
            _append_dialogue(dialogue_history, node, source_id)
            continue

        if node_type != "answer":
            raise ValueError(f"Unsupported timeline node type {node_type!r} in sample {sample_id}.")

        context_until = float(node.get("context_until", context_until))
        event_visible_start = float(node.get("visible_video_start", visible_video_start))
        event_visible_end = float(node.get("visible_video_end", visible_video_end))
        answer_timestamp = node.get("answer_timestamp", node.get("timestamp"))
        answer_timestamp = None if answer_timestamp is None else float(answer_timestamp)
        supervise = bool(node.get("supervise", True))
        mode = node.get("mode", "sequential")
        visible_in_history = bool(node.get("visible_in_history", mode == "sequential"))
        answer_texts = _answer_texts(node)

        if supervise:
            if mode == "multi_reference":
                event_id = str(node.get("answer_id", source_id))
                events.append(
                    AnswerEvent(
                        sample_id=sample_id,
                        event_id=event_id,
                        video=str(video),
                        visible_video_start=event_visible_start,
                        visible_video_end=event_visible_end,
                        context_until=context_until,
                        answer_timestamp=answer_timestamp,
                        dialogue_context=copy.deepcopy(dialogue_history),
                        target={
                            "role": node.get("role", "assistant"),
                            "answers": answer_texts,
                            "answer_id": event_id,
                            "mode": "multi_reference",
                        },
                        training_mode=training_mode,
                        source_dataset=None if source_dataset is None else str(source_dataset),
                        source_split=None if source_split is None else str(source_split),
                        turn_index=answer_turn_index,
                        frames=None if sample_frames is None else list(sample_frames),
                        frame_timestamps=None
                        if sample_frame_timestamps is None
                        else list(sample_frame_timestamps),
                    )
                )
            else:
                for answer_idx, answer_text in enumerate(answer_texts):
                    event_id = str(node.get("answer_id", source_id))
                    if len(answer_texts) > 1:
                        event_id = f"{event_id}_{answer_idx}"
                    events.append(
                        AnswerEvent(
                            sample_id=sample_id,
                            event_id=event_id,
                            video=str(video),
                            visible_video_start=event_visible_start,
                            visible_video_end=event_visible_end,
                            context_until=context_until,
                            answer_timestamp=answer_timestamp,
                            dialogue_context=copy.deepcopy(dialogue_history),
                            target={
                                "role": node.get("role", "assistant"),
                                "text": answer_text,
                                "answer_id": event_id,
                                "mode": mode,
                            },
                            training_mode=training_mode,
                            source_dataset=None if source_dataset is None else str(source_dataset),
                            source_split=None if source_split is None else str(source_split),
                            turn_index=answer_turn_index,
                            frames=None if sample_frames is None else list(sample_frames),
                            frame_timestamps=None
                            if sample_frame_timestamps is None
                            else list(sample_frame_timestamps),
                        )
                    )
                    if mode == "sequential" and visible_in_history:
                        dialogue_history.append(
                            {"role": node.get("role", "assistant"), "text": answer_text, "source_id": event_id}
                        )
                answer_turn_index += 1
                continue

        if visible_in_history and mode != "multi_reference":
            _append_dialogue(dialogue_history, {**node, "text": answer_texts[0]}, source_id)
        answer_turn_index += 1

    return events


def load_timeline_jsonl(path: str) -> List[Dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_answer_events(path: str) -> List[AnswerEvent]:
    events: List[AnswerEvent] = []
    for sample in load_timeline_jsonl(path):
        events.extend(timeline_to_events(sample))
    return events
