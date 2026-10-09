#!/usr/bin/env bash
set -euo pipefail

export SVBENCH_EVAL_MODE=dialogue
exec bash "$(dirname "$0")/run_svbench.sh" "$@"
