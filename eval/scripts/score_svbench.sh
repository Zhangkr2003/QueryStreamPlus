#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for the SVBench GPT judge.}"
SVBENCH_EVAL_MODE="${SVBENCH_EVAL_MODE:-dialogue}"
PREDICTIONS_JSONL="${PREDICTIONS_JSONL:-$project_root/outputs/evaluation/svbench/$SVBENCH_EVAL_MODE/querystream_plus.jsonl}"
JUDGE_OUTPUT_JSONL="${JUDGE_OUTPUT_JSONL:-$project_root/outputs/evaluation/svbench/$SVBENCH_EVAL_MODE/judgments.jsonl}"

extra_args=()
if [[ -n "${JUDGE_SUMMARY_JSON:-}" ]]; then extra_args+=(--summary-output "$JUDGE_SUMMARY_JSON"); fi
if [[ -n "${JUDGE_LIMIT_UNITS:-}" ]]; then extra_args+=(--limit-units "$JUDGE_LIMIT_UNITS"); fi
if [[ -n "${JUDGE_SAMPLE_SEED:-}" ]]; then extra_args+=(--sample-seed "$JUDGE_SAMPLE_SEED"); fi
if [[ -n "${ALLOW_INCOMPLETE:-}" ]]; then extra_args+=(--allow-incomplete); fi
if [[ -n "${OVERWRITE:-}" ]]; then extra_args+=(--overwrite); fi

python "$project_root/eval/scoring/gpt_judge.py" \
  --mode "$SVBENCH_EVAL_MODE" \
  --predictions "$PREDICTIONS_JSONL" \
  --judge-output "$JUDGE_OUTPUT_JSONL" \
  --judge-model "${JUDGE_MODEL:-gpt-4o}" \
  --workers "${JUDGE_WORKERS:-4}" \
  --max-retries "${JUDGE_MAX_RETRIES:-6}" \
  "${extra_args[@]}"
