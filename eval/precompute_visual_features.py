#!/usr/bin/env python3
"""Precompute visual caches for five benchmarks."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import av
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from querystream_plus.inference.engine import QueryStreamPlusConfig, QueryStreamPlusEngine
from eval import run_benchmark as benchmark_runner
from eval.scoring.svbench_protocol import official_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks", default="all",
        help="Comma-separated: streamingbench,ovobench,svbench,videomme,longvideobench, or all.",
    )
    parser.add_argument("--model-path", default=None, help="QueryStreamPlus-7B directory.")
    parser.add_argument("--siglip2-checkpoint", default=None)
    parser.add_argument("--visual-cache-dir", default=str(ROOT / "outputs" / "features"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="flash_attention_2",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-frames", type=int, default=1016)
    parser.add_argument("--ovo-max-frames", type=int, default=720)
    parser.add_argument("--stream-chunk-frames", type=int, default=16)
    parser.add_argument("--streaming-fps-schedule", choices=("adaptive", "fixed"), default="adaptive")
    parser.add_argument("--streamingbench-annotation")
    parser.add_argument("--streamingbench-video-dir")
    parser.add_argument(
        "--streamingbench-video-layout",
        choices=("auto", "source", "prechunked"),
        default="source",
    )
    parser.add_argument("--ovobench-annotation")
    parser.add_argument("--ovobench-video-dir")
    parser.add_argument("--ovobench-video-layout", default="auto")
    parser.add_argument("--svbench-annotation")
    parser.add_argument("--svbench-video-dir")
    parser.add_argument("--videomme-annotation")
    parser.add_argument("--videomme-video-dir")
    parser.add_argument("--longvideobench-annotation")
    parser.add_argument("--longvideobench-video-dir")
    parser.add_argument("--limit-videos", type=int)
    return parser.parse_args()


def duration(path: str) -> float:
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration / av.time_base)
        rate = float(stream.average_rate) if stream.average_rate else 1.0
        return float(stream.frames / rate)
    finally:
        container.close()


def require(args, benchmark, *names):
    missing = [name for name in names if not getattr(args, name)]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(f"{benchmark} requires {flags}")


def independent_specs(args, benchmark):
    if benchmark == "streamingbench":
        require(args, benchmark, "streamingbench_annotation", "streamingbench_video_dir")
        ns = SimpleNamespace(
            benchmark=benchmark, annotation=args.streamingbench_annotation,
            video_dir=args.streamingbench_video_dir,
            streamingbench_video_layout=args.streamingbench_video_layout,
        )
    elif benchmark == "ovobench":
        require(args, benchmark, "ovobench_annotation", "ovobench_video_dir")
        ns = SimpleNamespace(
            benchmark=benchmark, annotation=args.ovobench_annotation,
            video_dir=args.ovobench_video_dir, ovobench_video_layout=args.ovobench_video_layout,
        )
    elif benchmark == "videomme":
        require(args, benchmark, "videomme_annotation", "videomme_video_dir")
        ns = SimpleNamespace(
            benchmark=benchmark, annotation=args.videomme_annotation, video_dir=args.videomme_video_dir
        )
    else:
        raise ValueError(benchmark)
    seen = set()
    for row in benchmark_runner.load_rows(ns):
        path = str(Path(row["video"]).resolve())
        if benchmark == "streamingbench":
            row_fps = benchmark_runner.streaming_fps_for_timestamp(
                float(row["visible_end"]), args.streaming_fps_schedule, args.fps
            )
        else:
            row_fps = args.fps
        end = None if row.get("video_is_prefix", False) else row.get("visible_end")
        key = (path, float(row_fps), end)
        if key not in seen:
            seen.add(key)
            yield {"path": path, "fps": row_fps, "end_time": end, "turns": None}


def svbench_specs(args):
    require(args, "svbench", "svbench_annotation", "svbench_video_dir")
    sessions = defaultdict(lambda: {"path": None, "ends": []})
    for mode in ("dialogue", "streaming"):
        for row in official_rows(mode, args.svbench_annotation, args.svbench_video_dir):
            entry = sessions[(mode, row["session_id"])]
            entry["path"] = row["video"]
            if row.get("visible_end") is not None:
                entry["ends"].append(float(row["visible_end"]))
    for entry in sessions.values():
        if entry["path"] and entry["ends"]:
            yield {
                "path": str(Path(entry["path"]).resolve()),
                "fps": args.fps,
                "end_time": None,
                "turns": sorted(set(entry["ends"])),
            }


def longvideobench_specs(args):
    require(args, "longvideobench", "longvideobench_annotation", "longvideobench_video_dir")
    with open(args.longvideobench_annotation, encoding="utf-8") as handle:
        rows = json.load(handle)
    seen = set()
    for row in rows:
        path = str((Path(args.longvideobench_video_dir) / row["video_path"]).resolve())
        if path not in seen:
            seen.add(path)
            yield {"path": path, "fps": args.fps, "end_time": None, "turns": None}


def chunks_for_frames(frames, chunk_frames, turn_ends=None, video_duration=None):
    intervals = []
    if turn_ends:
        previous = 0
        for end in turn_ends:
            boundary = min(len(frames), max(2, int(round(end / max(video_duration, 1e-6) * len(frames)))))
            boundary -= boundary % 2
            if boundary <= previous:
                interval = [frames[max(0, boundary - 2)], frames[boundary - 1]]
            else:
                interval = frames[previous:boundary]
            intervals.append(interval)
            previous = boundary
    else:
        intervals = [frames]
    for interval in intervals:
        for start in range(0, len(interval), chunk_frames):
            chunk = list(interval[start : start + chunk_frames])
            if not chunk:
                continue
            if len(chunk) < 2:
                chunk.append(chunk[-1].copy())
            if len(chunk) % 2:
                chunk.append(chunk[-1].copy())
            yield chunk


def main():
    args = parse_args()
    selected = {value.strip() for value in args.benchmarks.split(",") if value.strip()}
    allowed = {"streamingbench", "ovobench", "svbench", "videomme", "longvideobench"}
    if selected == {"all"}:
        selected = allowed
    if not selected or not selected <= allowed:
        raise ValueError(f"Unknown benchmark selection: {sorted(selected - allowed)}")

    engine = QueryStreamPlusEngine(
        QueryStreamPlusConfig(
            checkpoint_path=args.model_path,
            siglip2_checkpoint=args.siglip2_checkpoint,
            device=args.device,
            attn_implementation=args.attn_implementation,
            semantic_device=args.device,
            fps=args.fps,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            max_frames=args.max_frames,
            stream_chunk_frames=args.stream_chunk_frames,
            visual_cache_dir=args.visual_cache_dir,
            visual_cache_mode="readwrite",
        )
    )
    chunk_frames = args.stream_chunk_frames - args.stream_chunk_frames % 2
    totals = defaultdict(int)
    all_specs = []
    for benchmark in sorted(selected):
        if benchmark in {"streamingbench", "ovobench", "videomme"}:
            specs = independent_specs(args, benchmark)
        elif benchmark == "svbench":
            specs = svbench_specs(args)
        else:
            specs = longvideobench_specs(args)
        for spec in specs:
            spec["benchmark"] = benchmark
            all_specs.append(spec)
    if args.limit_videos is not None:
        all_specs = all_specs[: args.limit_videos]

    for spec in tqdm(all_specs, desc="visual-cache videos"):
        if not Path(spec["path"]).is_file():
            print(f"[visual-cache][skip-missing] {spec['path']}", file=sys.stderr, flush=True)
            totals["missing"] += 1
            continue
        old_max = engine.config.max_frames
        engine.config.max_frames = args.ovo_max_frames if spec["benchmark"] == "ovobench" else args.max_frames
        try:
            frames = engine.load_video(spec["path"], fps=spec["fps"], end_time=spec["end_time"])
            video_seconds = duration(spec["path"]) if spec["turns"] else None
            for chunk in chunks_for_frames(frames, chunk_frames, spec["turns"], video_seconds):
                hit = engine.precompute_chunk(chunk)
                totals["feature_hits" if hit else "feature_writes"] += 1
                totals["chunks"] += 1
            totals[f"videos_{spec['benchmark']}"] += 1
        except Exception as exc:
            totals["errors"] += 1
            print(f"[visual-cache][skip-error] {spec['path']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        finally:
            engine.config.max_frames = old_max
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(json.dumps(dict(totals), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
