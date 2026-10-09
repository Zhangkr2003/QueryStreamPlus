#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_root/scripts/visual_budget.sh"
model_path="${MODEL_PATH:-$project_root/models/QueryStreamPlus-7B}"
router_checkpoint="${ROUTER_CHECKPOINT:-$model_path/router/router.pt}"
siglip2_path="${SIGLIP2_MODEL_PATH:-$project_root/models/siglip2-large-patch16-256}"
lora_adapter="${LORA_ADAPTER:-$model_path/adapter}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export HF_HOME="${HF_HOME:-$project_root/data/benchmarks}"
num_processes="${NUM_PROCESSES:-1}"
result_dir="${RESULT_DIR:-$project_root/outputs/evaluation/longvideobench}"
model_args="pretrained=${model_path},router_checkpoint=${router_checkpoint},siglip2_checkpoint=${siglip2_path},lora_adapter=${lora_adapter},attn_implementation=${ATTN_IMPLEMENTATION:-flash_attention_2},fps=${FPS:-1},min_pixels=$((256*28*28)),max_pixels=$((256*28*28)),max_num_frames=${MAX_FRAMES:-1016},stream_chunk_frames=${STREAM_CHUNK_FRAMES:-16},active_keep_rate=${keep_rate},qim_read_budget=${qim_read_budget},im_read_budget=${im_read_budget},qim_capacity=${qim_capacity},im_capacity=${im_capacity},active_buffer_capacity=${ACTIVE_BUFFER_CAPACITY:-2048},im_episode_budget=${IM_EPISODE_BUDGET:-128}"
if [[ -n "${VISUAL_CACHE_DIR:-}" ]]; then
  model_args+=",visual_cache_dir=${VISUAL_CACHE_DIR},visual_cache_mode=${VISUAL_CACHE_MODE:-readonly}"
fi
extra_args=()
if [[ -n "${LIMIT:-}" ]]; then extra_args+=(--limit "$LIMIT"); fi

python -m accelerate.commands.launch \
  --num_processes "$num_processes" \
  --main_process_port "${MAIN_PROCESS_PORT:-29511}" \
  -m lmms_eval \
  --model qwen2_5_vl_querystream_plus \
  --model_args "$model_args" \
  --tasks "${LONGVIDEOBENCH_TASK:-longvideobench_val_v}" \
  --batch_size 1 \
  --output_path "$result_dir/log" \
  "${extra_args[@]}"
