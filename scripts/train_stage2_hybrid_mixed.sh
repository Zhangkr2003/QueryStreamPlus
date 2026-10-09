#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$package_root/models/QueryStreamPlus-7B}"
SIGLIP2_MODEL_PATH="${SIGLIP2_MODEL_PATH:-$package_root/models/siglip2-large-patch16-256}"
TRAIN_JSONL="${TRAIN_JSONL:-$package_root/data/training/stage2.jsonl}"
MEDIA_ROOT="${MEDIA_ROOT:-$package_root/data/training/media}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$package_root/outputs/training}"
: "${ROUTER_CHECKPOINT:?Set ROUTER_CHECKPOINT to a Stage-I querystream_router_stepN.pt file.}"
train_jsonl="$TRAIN_JSONL"
[[ -f "$train_jsonl" ]] || {
  echo "Missing training manifest: $train_jsonl. Set TRAIN_JSONL to your prepared JSONL file." >&2
  exit 1
}

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
nproc_per_node="${NPROC_PER_NODE:-4}"
runtime_safety_args=(--distributed-timeout-seconds "${DISTRIBUTED_TIMEOUT_SECONDS:-3600}")
if [[ "$nproc_per_node" == "1" ]]; then
  runtime_safety_args+=(--skip-bad-samples)
  runtime_safety_args+=(--bad-sample-log "${BAD_SAMPLE_LOG:-$OUTPUT_ROOT/stage2_hybrid_mixed/bad_samples_rank0.jsonl}")
fi

cd "$package_root"
torchrun --standalone --nproc_per_node "$nproc_per_node" \
  --module querystream_plus.training.train \
  --training-stage joint \
  --memory-aware \
  --chunkwise-memory-training \
  --hybrid-memory-training \
  --stream-chunk-frames "${STREAM_CHUNK_FRAMES:-16}" \
  "${runtime_safety_args[@]}" \
  --mixed-visual-budgets 4096 6144 \
  --mixed-qim-read-budgets 1024 2048 \
  --mixed-im-read-budgets 1536 2560 \
  --router-init-checkpoint "$ROUTER_CHECKPOINT" \
  --model-path "$MODEL_PATH" \
  --train-jsonl "$train_jsonl" \
  --media-root "$MEDIA_ROOT" \
  --output-dir "$OUTPUT_ROOT/stage2_hybrid_mixed" \
  --device cuda \
  --torch-dtype bfloat16 \
  --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --vision-attn-implementation "${VISION_ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --semantic-device cuda \
  --siglip2-checkpoint "$SIGLIP2_MODEL_PATH" \
  --lora-r "${LORA_R:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}" \
  --lora-dropout "${LORA_DROPOUT:-0.05}" \
  --lr "${ROUTER_LR:-1e-4}" \
  --lora-lr "${LORA_LR:-1e-5}" \
  --epochs "${EPOCHS:-2}" \
  --fps "${FPS:-1}" \
  --max-frames "${MAX_FRAMES:-1016}" \
  --max-pixels "$((256 * 28 * 28))" \
  --max-visual-tokens 6144 \
  --keep-rate "${KEEP_RATE:-0.75}" \
  --qim-capacity "${QIM_CAPACITY:-2048}" \
  --im-capacity "${IM_CAPACITY:-3072}" \
  --active-buffer-capacity "${ACTIVE_BUFFER_CAPACITY:-2048}" \
  --im-episode-budget "${IM_EPISODE_BUDGET:-128}" \
  --qim-read-budget 2048 \
  --im-read-budget 2560 \
  --gradient-accumulation-steps "${GRAD_ACCUM:-16}" \
  --gradient-checkpointing \
  --warmup-ratio "${WARMUP_RATIO:-0.05}" \
  --save-every "${SAVE_EVERY:-1000}"
