#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$package_root/scripts/visual_budget.sh"
model_path="${MODEL_PATH:-$package_root/models/QueryStreamPlus-7B}"
siglip2_path="${SIGLIP2_MODEL_PATH:-$package_root/models/siglip2-large-patch16-256}"
router_checkpoint="${ROUTER_CHECKPOINT:-$model_path/router/router.pt}"
lora_adapter="${LORA_ADAPTER:-$model_path/adapter}"
session_json="${1:-${SESSION_JSON:-$package_root/examples/multiturn_session.json}}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"

python "$package_root/demo/infer_multiturn.py" \
  --session-json "$session_json" \
  --checkpoint-path "$model_path" \
  --router-checkpoint "$router_checkpoint" \
  --lora-adapter "$lora_adapter" \
  --device cuda \
  --device-map none \
  --torch-dtype bfloat16 \
  --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --semantic-device cuda \
  --siglip2-checkpoint "$siglip2_path" \
  --fps "${FPS:-1}" \
  --max-frames "${MAX_FRAMES:-1016}" \
  --max-pixels "$((256 * 28 * 28))" \
  --active-budget "${ACTIVE_BUDGET:-0}" \
  --active-keep-rate "$keep_rate" \
  --qim-capacity "$qim_capacity" \
  --im-capacity "$im_capacity" \
  --active-buffer-capacity "${ACTIVE_BUFFER_CAPACITY:-2048}" \
  --im-episode-budget "${IM_EPISODE_BUDGET:-128}" \
  --qim-read-budget "$qim_read_budget" \
  --im-read-budget "$im_read_budget" \
  --output-json "${OUTPUT_JSON:-$package_root/outputs/inference/querystream_plus_result.json}"
