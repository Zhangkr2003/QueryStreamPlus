#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local TimeChat-Online-7B or Qwen2.5-VL-7B-Instruct checkpoint.}"
: "${CLIP_PRETRAINED:?Set CLIP_PRETRAINED to the local OpenCLIP ViT-L-14 checkpoint.}"
: "${VIDEO:?Set VIDEO to a local video file.}"
: "${QUERY:?Set QUERY to the question.}"

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"

python "$package_root/demo/infer.py" \
  --video "$VIDEO" \
  --query "$QUERY" \
  --checkpoint-path "$MODEL_PATH" \
  --clip-pretrained "$CLIP_PRETRAINED" \
  --device cuda \
  --device-map none \
  --clip-device cuda \
  --fps "${FPS:-1}" \
  --print-qdp-stats \
  --print-token-stats \
  --output-json "${OUTPUT_JSON:-$package_root/outputs/querystream_result.json}"
