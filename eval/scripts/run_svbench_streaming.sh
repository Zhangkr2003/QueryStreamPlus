#!/usr/bin/env bash
set -euo pipefail

export SVBENCH_EVAL_MODE=streaming
exec bash "$(dirname "$0")/run_svbench.sh" "$@"
