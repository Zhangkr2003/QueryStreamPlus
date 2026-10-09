#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_root="${MODEL_ROOT:-$project_root/models}"

hf download KrZhang/QueryStreamPlus \
  --local-dir "$model_root/QueryStreamPlus-7B"

hf download google/siglip2-large-patch16-256 \
  --revision b8817f36154ccc84b692f29ef166a86406e8ddb1 \
  --local-dir "$model_root/siglip2-large-patch16-256"
