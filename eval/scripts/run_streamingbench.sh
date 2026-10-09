#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_root/scripts/visual_budget.sh"

model_path="${MODEL_PATH:-$project_root/models/QueryStreamPlus-7B}"
router_checkpoint="${ROUTER_CHECKPOINT:-$model_path/router/router.pt}"
siglip2_path="${SIGLIP2_MODEL_PATH:-$project_root/models/siglip2-large-patch16-256}"
lora_adapter="${LORA_ADAPTER:-$model_path/adapter}"
task_csv="${TASK_CSV:-$project_root/data/benchmarks/streamingbench/StreamingBench/Real_Time_Visual_Understanding.csv}"
video_dir="${VIDEO_DIR:-$project_root/data/benchmarks/streamingbench/Real-Time Visual Understanding}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"

extra_args=()
if [[ -n "${VISUAL_CACHE_DIR:-}" ]]; then
  extra_args+=(--visual-cache-dir "$VISUAL_CACHE_DIR" --visual-cache-mode "${VISUAL_CACHE_MODE:-readonly}")
fi
extra_args+=(--lora-adapter "$lora_adapter")
if [[ -n "${OVERWRITE:-}" ]]; then extra_args+=(--overwrite); fi
if [[ -n "${LIMIT:-}" ]]; then extra_args+=(--limit "$LIMIT"); fi
if [[ "${TIMING_FIRST:-1}" == "0" ]]; then extra_args+=(--no-timing-first); fi

python "$project_root/eval/run_benchmark.py" \
  --benchmark streamingbench \
  --annotation "$task_csv" \
  --video-dir "$video_dir" \
  --streamingbench-video-layout "${STREAMINGBENCH_VIDEO_LAYOUT:-source}" \
  --output-jsonl "${OUTPUT_JSONL:-$project_root/outputs/evaluation/streamingbench/querystream_plus.jsonl}" \
  --checkpoint-path "$model_path" \
  --router-checkpoint "$router_checkpoint" \
  --siglip2-checkpoint "$siglip2_path" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --fps "${FPS:-1}" \
  --streaming-fps-schedule "${STREAMING_FPS_SCHEDULE:-adaptive}" \
  --max-frames "${MAX_FRAMES:-1016}" \
  --stream-chunk-frames "${STREAM_CHUNK_FRAMES:-16}" \
  --active-keep-rate "$keep_rate" \
  --qim-read-budget "$qim_read_budget" \
  --im-read-budget "$im_read_budget" \
  --qim-capacity "$qim_capacity" \
  --im-capacity "$im_capacity" \
  --active-buffer-capacity "${ACTIVE_BUFFER_CAPACITY:-2048}" \
  --im-episode-budget "${IM_EPISODE_BUDGET:-128}" \
  "${extra_args[@]}"
