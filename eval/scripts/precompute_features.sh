#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

model_path="${MODEL_PATH:-$project_root/models/QueryStreamPlus-7B}"
siglip2_path="${SIGLIP2_MODEL_PATH:-$project_root/models/siglip2-large-patch16-256}"
visual_cache_dir="${VISUAL_CACHE_DIR:-$project_root/outputs/features}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"

args=(
  --benchmarks "${BENCHMARKS:-all}"
  --model-path "$model_path"
  --siglip2-checkpoint "$siglip2_path"
  --visual-cache-dir "$visual_cache_dir"
  --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}"
  --fps "${FPS:-1}"
  --max-frames "${MAX_FRAMES:-1016}"
  --ovo-max-frames "${OVO_MAX_FRAMES:-720}"
  --stream-chunk-frames "${STREAM_CHUNK_FRAMES:-16}"
  --streaming-fps-schedule "${STREAMING_FPS_SCHEDULE:-adaptive}"
  --streamingbench-video-layout "${STREAMINGBENCH_VIDEO_LAYOUT:-source}"
)

args+=(--streamingbench-annotation "${STREAMINGBENCH_ANNOTATION:-$project_root/data/benchmarks/streamingbench/StreamingBench/Real_Time_Visual_Understanding.csv}")
args+=(--streamingbench-video-dir "${STREAMINGBENCH_VIDEO_DIR:-$project_root/data/benchmarks/streamingbench/Real-Time Visual Understanding}")
args+=(--ovobench-annotation "${OVOBENCH_ANNOTATION:-$project_root/data/benchmarks/ovobench/ovo_bench_new.json}")
args+=(--ovobench-video-dir "${OVOBENCH_VIDEO_DIR:-$project_root/data/benchmarks/ovobench}")
args+=(--svbench-annotation "${SVBENCH_ANNOTATION:-$project_root/data/benchmarks/svbench}")
args+=(--svbench-video-dir "${SVBENCH_VIDEO_DIR:-$project_root/data/benchmarks/svbench}")
args+=(--videomme-annotation "${VIDEOMME_ANNOTATION:-$project_root/data/benchmarks/video_mme/test.parquet}")
args+=(--videomme-video-dir "${VIDEOMME_VIDEO_DIR:-$project_root/data/benchmarks/video_mme/videos}")
args+=(--longvideobench-annotation "${LONGVIDEOBENCH_ANNOTATION:-$project_root/data/benchmarks/longvideobench/lvb_val.json}")
args+=(--longvideobench-video-dir "${LONGVIDEOBENCH_VIDEO_DIR:-$project_root/data/benchmarks/longvideobench/videos}")

[[ -n "${LIMIT_VIDEOS:-}" ]] && args+=(--limit-videos "$LIMIT_VIDEOS")

python "$project_root/eval/precompute_visual_features.py" "${args[@]}"
