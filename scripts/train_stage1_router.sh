#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$package_root/models/QueryStreamPlus-7B}"
SIGLIP2_MODEL_PATH="${SIGLIP2_MODEL_PATH:-$package_root/models/siglip2-large-patch16-256}"
TRAIN_JSONL="${TRAIN_JSONL:-$package_root/data/training/stage1.jsonl}"
MEDIA_ROOT="${MEDIA_ROOT:-$package_root/data/training/media}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$package_root/outputs/training}"
train_jsonl="$TRAIN_JSONL"
[[ -f "$train_jsonl" ]] || {
  echo "Missing training manifest: $train_jsonl. Set TRAIN_JSONL to your prepared JSONL file." >&2
  exit 1
}
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
nproc_per_node="${NPROC_PER_NODE:-2}"

cd "$package_root"
torchrun --standalone --nproc_per_node "$nproc_per_node" --module querystream_plus.training.train \
  --training-stage router \
  --model-path "$MODEL_PATH" \
  --train-jsonl "$train_jsonl" \
  --media-root "$MEDIA_ROOT" \
  --output-dir "$OUTPUT_ROOT/stage1_router" \
  --device cuda \
  --torch-dtype bfloat16 \
  --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --vision-attn-implementation "${VISION_ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --semantic-device cuda \
  --siglip2-checkpoint "$SIGLIP2_MODEL_PATH" \
  --router-dim "${ROUTER_DIM:-768}" \
  --router-heads "${ROUTER_HEADS:-12}" \
  --router-spatial-window "${ROUTER_SPATIAL_WINDOW:-4}" \
  --router-temporal-window "${ROUTER_TEMPORAL_WINDOW:-8}" \
  --fps "${FPS:-1}" \
  --max-frames "${MAX_FRAMES:-1016}" \
  --max-pixels "$((256 * 28 * 28))" \
  --max-visual-tokens "${MAX_VISUAL_TOKENS:-8192}" \
  --keep-rate 0.5 \
  --gradient-accumulation-steps "${GRAD_ACCUM:-8}" \
  --gradient-checkpointing \
  --save-every "${SAVE_EVERY:-1000}"
