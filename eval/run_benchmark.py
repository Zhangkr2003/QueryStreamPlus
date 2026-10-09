#!/usr/bin/env python3
"""Unified passive benchmark evaluation for QueryStream++."""

from __future__ import annotations

import argparse
import ast
import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import av
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from querystream_plus.inference.engine import QueryStreamPlusConfig, QueryStreamPlusEngine
from eval.scoring.svbench_protocol import official_rows


VIDEOMME_PROMPT = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{question}
Options: {options}
The best answer is:"""

STREAMINGBENCH_PROMPT = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {question}

Options:
{options}

The best option is:"""

OVOBENCH_MC_PROMPT = """Question: {question}
Options:
{options}

Respond only with the letter corresponding to your chosen option (e.g., A, B, C).
Do not include any additional text or explanation in your response."""

OVOBENCH_REC_PROMPT = """You're watching a video in which people may perform a certain type of action repetively.
The person performing this kind of action are referred to as 'they' in the following statement.
You're task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one.
Now, answer the following question: How many times did they {activity}?
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response."""

OVOBENCH_SSR_PROMPT = """You're watching a tutorial video which contain a sequential of steps.
The following is one step from the whole procedures:
{step}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response."""

OVOBENCH_CRR_PROMPT = """You're responsible of answering questions based on the video content.
The following question are relevant to the latest frames, i.e. the end of the video.
{question}
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response."""

GENERIC_MC_PROMPT = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter of the correct option.
Question: {question}
Options:
{options}
The best answer is:"""


def parse_args():
    parser = argparse.ArgumentParser(description="QueryStream++ passive benchmark evaluation.")
    parser.add_argument(
        "--benchmark",
        choices=("videomme", "streamingbench", "ovobench", "svbench", "generic"),
        required=True,
    )
    parser.add_argument("--annotation", required=True, help="Video-MME parquet, StreamingBench CSV, or generic JSONL.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument(
        "--svbench-eval-mode",
        choices=("dialogue", "streaming"),
        default="dialogue",
        help="Official SVBench Dialogue or Streaming protocol.",
    )
    parser.add_argument(
        "--streamingbench-video-layout",
        choices=("auto", "source", "prechunked"),
        default="auto",
        help=(
            "StreamingBench media layout. prechunked resolves "
            "sample_xxx/tmp_internvideo/video_0_<timestamp>.mp4; source uses "
            "sample_xxx/video.mp4; auto prefers an existing pre-cut prefix."
        ),
    )
    parser.add_argument(
        "--ovobench-video-layout",
        choices=("auto", "source", "chunked"),
        default="auto",
        help=(
            "OVO-Bench media layout. source resolves item['video']; chunked resolves "
            "<id>.mp4 and <id>_<test_index>.mp4; auto prefers an existing chunk."
        ),
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--router-checkpoint", default=None)
    parser.add_argument("--lora-adapter", default=None)
    parser.add_argument("--siglip2-checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="flash_attention_2",
        help="Attention backend (default: flash_attention_2).",
    )
    parser.add_argument("--semantic-device", default="cuda")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--streaming-fps-schedule",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help="StreamingBench adaptive uses 1 FPS <=300s, 0.5 FPS <=600s, then 0.2 FPS.",
    )
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-frames", type=int, default=1016)
    parser.add_argument("--stream-chunk-frames", type=int, default=16)
    parser.add_argument("--active-keep-rate", type=float, default=0.75)
    parser.add_argument("--active-budget", type=int, default=0)
    parser.add_argument("--qim-read-budget", type=int, default=1024)
    parser.add_argument("--im-read-budget", type=int, default=1536)
    parser.add_argument("--qim-capacity", type=int, default=2048)
    parser.add_argument("--im-capacity", type=int, default=3072)
    parser.add_argument("--active-buffer-capacity", type=int, default=2048)
    parser.add_argument("--im-episode-budget", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--visual-cache-dir", default=None)
    parser.add_argument(
        "--visual-cache-mode", choices=("readwrite", "readonly", "refresh"), default="readwrite"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--timing-first",
        dest="timing_first",
        action="store_true",
        default=True,
        help="Print each stage time immediately for only the first inferred sample (default).",
    )
    parser.add_argument(
        "--no-timing-first",
        dest="timing_first",
        action="store_false",
        help="Disable first-sample stage timing prints.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output JSONL and evaluate every sample again.",
    )
    return parser.parse_args()


def answer_letter(text: str) -> str:
    match = re.search(r"\b([A-Z])\b", str(text).upper())
    return "" if match is None else match.group(1)


def options_text(options) -> str:
    values = options.tolist() if hasattr(options, "tolist") else list(options)
    result = []
    for index, value in enumerate(values):
        value = str(value)
        if re.match(r"^[A-Z][\.)]\s*", value):
            result.append(value)
        else:
            result.append(f"{chr(65 + index)}. {value}")
    return "\n".join(result)


def build_benchmark_prompt(benchmark: str, row: dict) -> str:
    """Build the prompt prescribed by each benchmark's official evaluator."""
    question = str(row["question"])
    options = row.get("options")

    # Official SVBench evaluation sends the annotated question as-is. The
    # official LongVideoBench prompt is constructed separately by lmms-eval.
    if benchmark == "svbench" or (options is None and benchmark != "ovobench"):
        return question

    rendered_options = None if options is None else options_text(options)
    if benchmark == "videomme":
        return VIDEOMME_PROMPT.format(question=question, options=rendered_options)
    if benchmark == "streamingbench":
        return STREAMINGBENCH_PROMPT.format(question=question, options=rendered_options)
    if benchmark == "ovobench":
        metadata = row.get("metadata", {})
        task = str(metadata.get("task", metadata.get("item", {}).get("task", "")))
        if options is not None:
            return OVOBENCH_MC_PROMPT.format(
                question=question,
                options=rendered_options.replace("\n", "; ") + ";",
            )
        item = metadata["item"]
        test = item["test_info"][int(metadata["test_index"])]
        if task == "REC":
            return OVOBENCH_REC_PROMPT.format(activity=item["activity"])
        if task == "SSR":
            return OVOBENCH_SSR_PROMPT.format(step=test["step"])
        if task == "CRR":
            return OVOBENCH_CRR_PROMPT.format(question=item["question"])
        raise ValueError(f"Unsupported OVO-Bench task for prompting: {task}")
    if options is not None:
        return GENERIC_MC_PROMPT.format(question=question, options=rendered_options)
    return question


def time_to_seconds(value) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    parts = [float(part) for part in str(value).split(":")]
    total = 0.0
    for part in parts:
        total = total * 60.0 + part
    return total


def streaming_fps_for_timestamp(timestamp: float, schedule: str, fixed_fps: float) -> float:
    if schedule == "fixed":
        return fixed_fps
    if timestamp <= 300:
        return 1.0
    if timestamp <= 600:
        return 0.5
    return 0.2


def resolve_streamingbench_video(args, session: str, timestamp: float):
    sample_dir = Path(args.video_dir) / session
    source_path = sample_dir / "video.mp4"
    clip_dir = sample_dir / "tmp_internvideo"
    timestamp = float(timestamp)
    timestamp_label = str(int(timestamp)) if timestamp.is_integer() else f"{timestamp:g}"
    direct_clip = clip_dir / f"video_0_{timestamp_label}.mp4"
    layout = getattr(args, "streamingbench_video_layout", "auto")

    def existing_clip():
        if direct_clip.is_file():
            return direct_clip
        # Accept equivalent timestamp formatting.
        for candidate in sorted(clip_dir.glob("video_0_*.mp4")):
            end_label = candidate.stem.removeprefix("video_0_")
            try:
                if abs(float(end_label) - timestamp) < 1e-6:
                    return candidate
            except ValueError:
                continue
        return None

    if layout != "source":
        clip = existing_clip()
        if clip is not None:
            return str(clip), True
        if layout == "prechunked":
            raise FileNotFoundError(
                f"Missing StreamingBench pre-cut prefix for {session} at {timestamp:g}s. "
                f"Expected {direct_clip}."
            )
    if source_path.is_file():
        return str(source_path), False
    raise FileNotFoundError(
        f"Cannot resolve StreamingBench video for {session} at {timestamp:g}s. "
        f"Checked pre-cut prefix {direct_clip} and source video {source_path}."
    )


def resolve_ovobench_video(args, item, test_index=None):
    root = Path(args.video_dir)
    source_path = root / item["video"]
    chunk_name = f"{item['id']}.mp4" if test_index is None else f"{item['id']}_{test_index}.mp4"
    chunk_path = root / chunk_name
    layout = getattr(args, "ovobench_video_layout", "auto")
    if layout == "chunked":
        return str(chunk_path), True
    if layout == "source":
        return str(source_path), False
    if chunk_path.is_file():
        return str(chunk_path), True
    if source_path.is_file():
        return str(source_path), False
    raise FileNotFoundError(
        "Cannot resolve OVO-Bench video in auto mode. Checked official chunk "
        f"{chunk_path} and source video {source_path}."
    )


def load_rows(args):
    if args.benchmark == "videomme":
        import pandas as pd

        frame = pd.read_parquet(args.annotation)
        for row in frame.to_dict("records"):
            yield {
                "sample_id": str(row["question_id"]),
                "session_id": str(row["videoID"]),
                "video": str(Path(args.video_dir) / f"{row['videoID']}.mp4"),
                "visible_end": None,
                "question": row["question"],
                "options": row["options"].tolist() if hasattr(row["options"], "tolist") else list(row["options"]),
                "answer": row["answer"],
                "metadata": row,
            }
        return
    if args.benchmark == "ovobench":
        with open(args.annotation, encoding="utf-8") as handle:
            items = json.load(handle)
        for item in items:
            task = item["task"]
            if task in {"EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"}:
                video, video_is_prefix = resolve_ovobench_video(args, item)
                yield {
                    "sample_id": str(item["id"]),
                    "session_id": str(item["video"]),
                    "video": video,
                    "video_is_prefix": video_is_prefix,
                    "independent_prefix": True,
                    "visible_end": float(item["realtime"]),
                    "question": item["question"],
                    "options": item["options"],
                    "answer": chr(65 + int(item["gt"])),
                    "metadata": item,
                }
                continue
            for index, test in enumerate(item["test_info"]):
                video, video_is_prefix = resolve_ovobench_video(args, item, index)
                if task == "REC":
                    question = (
                        "Count how many times people perform the following action in total. "
                        f"One complete motion counts as one. Action: {item['activity']} "
                        "Respond with only a single number."
                    )
                    answer = str(test["count"])
                elif task == "SSR":
                    question = (
                        "Determine whether the person is currently performing this tutorial step: "
                        f"{test['step']} Respond only with Yes or No."
                    )
                    answer = "Yes" if int(test["type"]) == 1 else "No"
                elif task == "CRR":
                    question = (
                        f"Question: {item['question']} Determine whether the visual content near the "
                        "current end of the video provides enough information to answer it. Respond only Yes or No."
                    )
                    answer = "Yes" if int(test["type"]) == 1 else "No"
                else:
                    raise ValueError(f"Unsupported OVO-Bench task: {task}")
                yield {
                    "sample_id": f"{item['id']}_{index}",
                    "session_id": str(item["video"]),
                    "video": video,
                    "video_is_prefix": video_is_prefix,
                    "independent_prefix": True,
                    "visible_end": float(test["realtime"]),
                    "question": question,
                    "options": None,
                    "answer": answer,
                    "metadata": {"task": task, "item": item, "test_index": index},
                }
        return
    if args.benchmark == "streamingbench":
        import pandas as pd

        frame = pd.read_csv(args.annotation)
        for row in frame.to_dict("records"):
            qid = str(row["question_id"])
            session = f"sample_{qid.split('_')[-2]}"
            visible_end = time_to_seconds(row["time_stamp"])
            video, video_is_prefix = resolve_streamingbench_video(args, session, visible_end)
            yield {
                "sample_id": qid,
                "session_id": session,
                "video": video,
                "video_is_prefix": video_is_prefix,
                "independent_prefix": True,
                "visible_end": visible_end,
                "question": row["question"],
                "options": ast.literal_eval(row["options"]) if isinstance(row["options"], str) else row["options"],
                "answer": row["answer"],
                "metadata": row,
            }
        return
    if args.benchmark == "svbench":
        yield from official_rows(args.svbench_eval_mode, args.annotation, args.video_dir)
        return
    with open(args.annotation, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            yield {
                "sample_id": str(row.get("id", line_number)),
                "session_id": str(row.get("session_id", row.get("video"))),
                "video": str(Path(args.video_dir) / row["video"]),
                "visible_end": None if row.get("visible_end") is None else float(row["visible_end"]),
                "question": row["question"],
                "options": row.get("options"),
                "answer": row.get("answer"),
                "metadata": row.get("metadata", {}),
            }


def existing_ids(path: Path):
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[resume] ignoring malformed JSONL line {line_number} in {path}: {exc}",
                    file=sys.stderr,
                )
                continue
            if "sample_id" not in payload:
                print(
                    f"[resume] ignoring JSONL line {line_number} without sample_id in {path}",
                    file=sys.stderr,
                )
                continue
            completed.add(str(payload["sample_id"]))
    return completed


def summarize_saved_results(path: Path):
    """Aggregate the latest saved prediction for every sample ID."""
    latest = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"[summary] ignoring malformed JSONL line {line_number} in {path}",
                        file=sys.stderr,
                    )
                    continue
                if row.get("sample_id") is not None:
                    latest[str(row["sample_id"])] = row
    grouped = defaultdict(lambda: [0, 0])
    scored = 0
    correct = 0
    for row in latest.values():
        if row.get("correct") is None:
            continue
        category = str(row.get("category", "overall"))
        is_correct = bool(row["correct"])
        grouped[category][0] += 1
        grouped[category][1] += int(is_correct)
        scored += 1
        correct += int(is_correct)
    group_accuracies = [category_correct / count for count, category_correct in grouped.values() if count]
    return {
        "processed": len(latest),
        "scored": scored,
        "correct": correct,
        "accuracy": None if scored == 0 else correct / scored,
        "macro_group_accuracy": (
            None if not group_accuracies else sum(group_accuracies) / len(group_accuracies)
        ),
        "output": str(path),
        "groups": {
            category: {
                "count": count,
                "correct": category_correct,
                "accuracy": category_correct / count,
            }
            for category, (count, category_correct) in sorted(grouped.items())
            if count
        },
    }


def passive_prediction(response: str, options, answer) -> str:
    if options is not None:
        return answer_letter(response)
    expected = str(answer).strip().lower() if answer is not None else ""
    text = str(response).strip()
    if expected in {"yes", "no"}:
        match = re.search(r"\b(yes|no|y|n)\b", text.lower())
        if match is None:
            return text
        return "Yes" if match.group(1) in {"yes", "y"} else "No"
    if re.fullmatch(r"\d+", expected):
        match = re.search(r"\d+", text)
        return text if match is None else match.group(0)
    return text


def video_duration(path: str) -> float:
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration / av.time_base)
        fps = float(stream.average_rate) if stream.average_rate else 1.0
        return float(stream.frames / fps)
    finally:
        container.close()


def main():
    args = parse_args()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and args.overwrite:
        output_path.unlink()
    completed = existing_ids(output_path)

    rows = list(load_rows(args))
    if args.limit is not None:
        rows = rows[: args.limit]
    # Preserve causal order for session-based data.
    if args.benchmark in {"generic", "svbench"}:
        rows.sort(
            key=lambda row: (
                row["session_id"],
                row.get(
                    "sequence_index",
                    float("inf") if row["visible_end"] is None else row["visible_end"],
                ),
                row["sample_id"],
            )
        )
    original_row_count = len(rows)
    # Skip completed independent questions before model loading.
    can_skip_upfront = all(bool(row.get("independent_prefix", False)) for row in rows)
    if completed and can_skip_upfront:
        rows = [row for row in rows if str(row["sample_id"]) not in completed]
    elif completed and args.benchmark == "svbench":
        # Replay partially completed sessions to rebuild memory.
        session_rows = defaultdict(list)
        for row in rows:
            session_rows[str(row["session_id"])].append(row)
        complete_sessions = {
            session_id
            for session_id, values in session_rows.items()
            if all(str(value["sample_id"]) in completed for value in values)
        }
        if complete_sessions:
            rows = [row for row in rows if str(row["session_id"]) not in complete_sessions]
    skipped_count = original_row_count - len(rows)
    print(
        json.dumps(
            {
                "output_jsonl": str(output_path),
                "annotation_rows": original_row_count,
                "completed_ids_in_output": len(completed),
                "skipped_before_inference": skipped_count,
                "remaining_rows": len(rows),
                "overwrite": args.overwrite,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not rows:
        print("All annotation rows are already present in the result JSONL.", flush=True)
        print(json.dumps(summarize_saved_results(output_path), ensure_ascii=False, indent=2))
        return
    config = QueryStreamPlusConfig(
        checkpoint_path=args.checkpoint_path,
        router_checkpoint=args.router_checkpoint,
        lora_adapter=args.lora_adapter,
        siglip2_checkpoint=args.siglip2_checkpoint,
        device=args.device,
        attn_implementation=args.attn_implementation,
        semantic_device=args.semantic_device,
        fps=args.fps,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_frames=args.max_frames,
        stream_chunk_frames=args.stream_chunk_frames,
        active_keep_rate=args.active_keep_rate,
        active_budget=args.active_budget,
        qim_read_budget=args.qim_read_budget,
        im_read_budget=args.im_read_budget,
        qim_capacity=args.qim_capacity,
        im_capacity=args.im_capacity,
        active_buffer_capacity=args.active_buffer_capacity,
        im_episode_budget=args.im_episode_budget,
        max_new_tokens=args.max_new_tokens,
        visual_cache_dir=args.visual_cache_dir,
        visual_cache_mode=args.visual_cache_mode,
    )
    engine = QueryStreamPlusEngine(config)
    current_session = None
    full_frames = None
    previous_boundary = 0
    previous_visible_end = 0.0
    current_duration = None
    current_video_is_prefix = False
    current_video_path = None
    timing_emitted = False

    for row in tqdm(rows, desc=args.benchmark):
        was_completed = row["sample_id"] in completed
        independent_prefix = bool(row.get("independent_prefix", False))
        sampling_fps = args.fps
        if independent_prefix and was_completed:
            # Independent questions need no state replay.
            continue
        timing_this_row = args.timing_first and not timing_emitted
        if timing_this_row:
            engine.enable_stage_timing(str(row["sample_id"]))
        if independent_prefix:
            interval = None
            full_frames = None
            engine.reset()
            current_session = None
            current_video_is_prefix = bool(row.get("video_is_prefix", False))
            row_fps = (
                streaming_fps_for_timestamp(
                    float(row["visible_end"]), args.streaming_fps_schedule, args.fps
                )
                if args.benchmark == "streamingbench"
                else args.fps
            )
            sampling_fps = row_fps
            prefix_end = None if current_video_is_prefix else row["visible_end"]
            video_started = time.perf_counter()
            try:
                interval = engine.load_video(row["video"], fps=row_fps, end_time=prefix_end)
            except av.error.InvalidDataError as exc:
                # Fall back to the source video when a pre-cut clip is invalid.
                source_video = Path(args.video_dir) / str(row["session_id"]) / "video.mp4"
                if not (
                    args.benchmark == "streamingbench"
                    and current_video_is_prefix
                    and source_video.is_file()
                ):
                    raise
                print(
                    f"[streamingbench][decode-fallback] invalid pre-cut video {row['video']}; "
                    f"using source {source_video} up to {float(row['visible_end']):g}s: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                interval = engine.load_video(
                    str(source_video), fps=row_fps, end_time=float(row["visible_end"])
                )
                row["decode_fallback_video"] = str(source_video)
            if timing_this_row:
                print(
                    f"[timing][sample={row['sample_id']}][stage=video_decode_resize] "
                    f"{time.perf_counter() - video_started:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )
        elif row["session_id"] != current_session:
            # Release the previous video's buffers.
            same_source_video = (
                full_frames is not None
                and current_video_path == row["video"]
                and not bool(row.get("video_is_prefix", False))
            )
            if not same_source_video:
                full_frames = None
            engine.reset()
            gc.collect()
            current_session = row["session_id"]
            current_video_is_prefix = bool(row.get("video_is_prefix", False))
            current_video_path = row["video"]
            if not same_source_video:
                current_duration = None
            if not current_video_is_prefix and not same_source_video:
                video_started = time.perf_counter()
                full_frames = engine.load_video(row["video"])
                if timing_this_row:
                    print(
                        f"[timing][sample={row['sample_id']}][stage=video_decode_resize] "
                        f"{time.perf_counter() - video_started:.3f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                current_duration = video_duration(row["video"])
            previous_boundary = 0
            previous_visible_end = 0.0
        elif bool(row.get("video_is_prefix", False)) != current_video_is_prefix:
            raise RuntimeError(
                f"Mixed source/chunked layouts within OVO-Bench session {current_session}."
            )
        query = build_benchmark_prompt(args.benchmark, row)
        query_only = False
        if independent_prefix:
            pass
        elif row["visible_end"] is None:
            # Evaluate untimed questions independently.
            engine.reset()
            interval = full_frames
        elif current_video_is_prefix:
            # Keep only the unseen interval from the current OVO-Bench prefix.
            video_started = time.perf_counter()
            prefix_frames = engine.load_video(row["video"])
            if timing_this_row:
                print(
                    f"[timing][sample={row['sample_id']}][stage=video_decode_resize] "
                    f"{time.perf_counter() - video_started:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )
            prefix_duration = video_duration(row["video"])
            visible_end = min(float(row["visible_end"]), prefix_duration)
            start_boundary = int(
                round(min(previous_visible_end, visible_end) / max(prefix_duration, 1e-6) * len(prefix_frames))
            )
            end_boundary = int(
                round(visible_end / max(prefix_duration, 1e-6) * len(prefix_frames))
            )
            start_boundary = min(len(prefix_frames) - 2, max(0, start_boundary))
            start_boundary -= start_boundary % 2
            end_boundary = min(len(prefix_frames), max(2, end_boundary))
            end_boundary -= end_boundary % 2
            if end_boundary <= start_boundary:
                interval = prefix_frames[max(0, end_boundary - 2) : end_boundary]
            else:
                interval = prefix_frames[start_boundary:end_boundary]
            previous_visible_end = max(previous_visible_end, float(row["visible_end"]))
        else:
            # Map timestamps after uniform frame sampling.
            boundary = min(
                len(full_frames),
                max(2, int(round(float(row["visible_end"]) / max(current_duration, 1e-6) * len(full_frames)))),
            )
            boundary -= boundary % 2
            if bool(row.get("reuse_visible_state", False)):
                query_only = True
                interval = None
            elif boundary <= previous_boundary and args.benchmark == "svbench":
                # Reuse state when sampling exposes no new frame.
                query_only = True
                interval = None
            elif boundary <= previous_boundary:
                interval = [full_frames[max(0, boundary - 2)], full_frames[boundary - 1]]
            else:
                interval = full_frames[previous_boundary:boundary]
            previous_boundary = max(previous_boundary, boundary)
        result = (
            engine.answer_without_ingest(query)
            if query_only
            else engine.answer_interval(interval, query)
        )
        if timing_this_row:
            engine.disable_stage_timing()
            timing_emitted = True
            print(
                f"[timing][sample={row['sample_id']}] finished; timing disabled for later samples",
                file=sys.stderr,
                flush=True,
            )
        result["sampling_fps"] = sampling_fps
        result["independent_prefix"] = independent_prefix
        if row.get("decode_fallback_video"):
            result["decode_fallback_video"] = row["decode_fallback_video"]
        # Replay completed stateful turns to rebuild memory.
        if was_completed:
            continue
        prediction = passive_prediction(result["answer"], row["options"], row["answer"])
        uses_external_judge = (
            args.benchmark == "svbench" and row["metadata"].get("svbench_eval_mode") is not None
        )
        correct = (
            None
            if row["answer"] is None or uses_external_judge
            else prediction == str(row["answer"]).strip()
        )
        if args.benchmark == "videomme":
            category = str(row["metadata"].get("duration", "unknown"))
        elif args.benchmark == "streamingbench":
            category = str(row["metadata"].get("task_type", "unknown"))
        elif args.benchmark == "ovobench":
            category = str(
                row["metadata"].get(
                    "task", row["metadata"].get("item", {}).get("task", "unknown")
                )
            )
        elif args.benchmark == "svbench":
            category = str(
                row["metadata"].get(
                    "dependency", row["metadata"].get("svbench_eval_mode", "unknown")
                )
            )
        else:
            category = "generic"
        payload = {
            "benchmark": args.benchmark,
            "sample_id": row["sample_id"],
            "session_id": row["session_id"],
            "video": row["video"],
            "visible_end": row["visible_end"],
            "question": row["question"],
            "options": row["options"],
            "answer": row["answer"],
            "response": result.pop("answer"),
            "prediction": prediction,
            "correct": correct,
            "category": category,
        }
        if args.benchmark == "svbench" and row["metadata"].get("svbench_eval_mode"):
            payload.update(
                {
                    "evaluation_mode": row["metadata"]["svbench_eval_mode"],
                    "source_video_id": row["metadata"]["source_video_id"],
                    "chain_index": row["metadata"].get("chain_index"),
                    "turn_index": row["metadata"].get("turn_index"),
                    "temporal_relationship": row["metadata"].get(
                        "relationship_from_previous", []
                    ),
                }
            )
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    summary = summarize_saved_results(output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
