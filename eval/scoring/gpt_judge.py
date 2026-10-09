#!/usr/bin/env python3
"""LLM-judge scoring for official SVBench Dialogue and Streaming outputs."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


METRICS = [
    "Semantic Accuracy",
    "Contextual Coherence",
    "Logical Consistency",
    "Temporal Understanding",
    "Informational Completeness",
    "Overall Evaluation",
]
EXPECTED = {
    "dialogue": {"rows": 7850, "units": 1822, "sessions": 200},
    "streaming": {"rows": 23635, "units": 1000, "sessions": 1000},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Official SVBench GPT judge.")
    parser.add_argument("--mode", choices=("dialogue", "streaming"), required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--judge-output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument(
        "--limit-units",
        type=int,
        default=None,
        help="Judge a deterministic random subset, then resume the remainder in a later run.",
    )
    parser.add_argument("--sample-seed", type=int, default=20260824)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
    return rows


def group_predictions(rows, mode):
    groups = defaultdict(list)
    sessions = set()
    for row in rows:
        if row.get("evaluation_mode") != mode:
            raise ValueError(
                f"Prediction {row.get('sample_id')} is not an official SVBench {mode} row."
            )
        sessions.add(str(row["session_id"]))
        if mode == "dialogue":
            unit_id = f"{row['source_video_id']}:chain:{int(row['chain_index']):04d}"
        else:
            unit_id = str(row["session_id"])
        groups[unit_id].append(row)
    for values in groups.values():
        values.sort(key=lambda row: int(row.get("turn_index", 0)))
    return dict(groups), sessions


def validate_coverage(rows, groups, sessions, mode, allow_incomplete):
    expected = EXPECTED[mode]
    actual = {"rows": len(rows), "units": len(groups), "sessions": len(sessions)}
    print(json.dumps({"mode": mode, "expected": expected, "actual": actual}, ensure_ascii=False))
    if not allow_incomplete and actual != expected:
        raise RuntimeError(
            "SVBench output is incomplete; refusing to report an official score. "
            "Resume inference or pass --allow-incomplete only for debugging."
        )


def _turn_text(rows, mode):
    blocks = []
    for index, row in enumerate(rows, 1):
        relationship = row.get("temporal_relationship", []) if mode == "streaming" else []
        relationship_text = (
            f"\nTemporal relationship from previous turn: {relationship}" if relationship else ""
        )
        blocks.append(
            f"Turn {index} (visible timestamp: {row.get('visible_end')}s):\n"
            f"Question: {row.get('question', '')}\n"
            f"Ground truth: {row.get('answer', '')}\n"
            f"Model response: {row.get('response', '')}{relationship_text}"
        )
    return "\n\n".join(blocks)


def make_prompt(unit_id, rows, mode):
    setting = (
        "questions at the same timestamp within a streaming video"
        if mode == "dialogue"
        else "temporally linked questions at different timestamps within a streaming video"
    )
    return f"""You are an evaluation expert for SVBench. Evaluate the model responses to
{setting}. Compare them with the ground-truth answers while considering the complete
multi-turn context. Follow the released SVBench multidimensional judge criteria.

1. Semantic Accuracy evaluates factual and semantic correctness. Score 10 for completely
accurate answers with no apparent errors; 7-9 for mostly accurate answers with minor detail
errors; 4-6 when several errors remain but most content is conveyed; 1-3 when only a small
part is accurate; and 0 for completely inaccurate or unrelated answers.

2. Contextual Coherence evaluates relevance and continuity across sequential questions and
answers. Score 10 for highly coherent dialogue; 7-9 for mostly coherent dialogue with minor
transition issues; 4-6 for partially disjointed dialogue; 1-3 for poor coherence; and 0 for
completely unrelated or independent content.

3. Logical Consistency evaluates progression and contradictions. Score 10 for fully logical
and consistent answers; 7-9 for a few minor inconsistencies; 4-6 for several logical issues
while remaining understandable; 1-3 for largely unreasonable answers; and 0 for completely
illogical answers that contradict the video content.

4. Temporal Understanding evaluates event order, timing, and temporal/causal relationships.
Score 10 for a fully correct timeline; 7-9 for minor temporal errors; 4-6 for partial temporal
understanding with significant omissions; 1-3 for little correct temporal understanding; 0
for complete temporal failure; and -1 only when the sequence does not involve temporal
understanding at all.

5. Informational Completeness evaluates whether all relevant elements are conveyed. Score
10 for fully comprehensive answers; 7-9 for mostly comprehensive answers; 4-6 for partially
informative but incomplete answers; 1-3 for largely incomplete answers; and 0 when no useful
information is provided.

6. Overall Evaluation is holistic: 1-2 is irrelevant or factually incorrect, 3-4 is low
quality, 5-6 is moderate quality, 7-8 is high quality, and 9-10 is excellent. Use 0 only for
a complete failure.

Unit ID: {unit_id}

{_turn_text(rows, mode)}

Return only one JSON object with exactly these keys. Each value must contain an integer
\"score\" in [0, 10] and a concise \"comments\" string. Temporal Understanding may be -1
only under the non-temporal rule above:
{json.dumps({metric: {"score": None, "comments": ""} for metric in METRICS}, ensure_ascii=False)}
"""


def parse_judge_json(text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            raise
        payload = json.loads(match.group(0))
    scores = {}
    for metric in METRICS:
        value = payload.get(metric)
        score = value.get("score") if isinstance(value, dict) else value
        score = int(score)
        minimum = -1 if metric == "Temporal Understanding" else 0
        if not minimum <= score <= 10:
            raise ValueError(f"Invalid {metric} score: {score}")
        scores[metric] = score
        if not isinstance(value, dict):
            payload[metric] = {"score": score, "comments": ""}
    return payload, scores


def call_judge(api_key, model, prompt, max_retries):
    endpoint = "https://api.openai.com/v1/chat/completions"
    request_payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(request_payload).encode("utf-8")
    for attempt in range(max_retries):
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = json.loads(response.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            parsed, scores = parse_judge_json(content)
            usage = raw.get("usage", {})
            return content, parsed, scores, usage
        except Exception as exc:
            if attempt + 1 == max_retries:
                raise RuntimeError(f"Judge failed after {max_retries} attempts: {exc}") from exc
            time.sleep(min(2 ** attempt, 30))


def completed_units(path):
    if not path.is_file():
        return set()
    return {str(row["unit_id"]) for row in read_jsonl(path) if "unit_id" in row}


def validate_existing_judge(path, model, mode):
    if not path.is_file():
        return
    rows = read_jsonl(path)
    incompatible = [
        row.get("unit_id")
        for row in rows
        if row.get("judge_model") != model or row.get("mode") != mode
    ]
    if incompatible:
        raise RuntimeError(
            f"Judge output already contains a different model or mode (first: {incompatible[0]}). "
            "Use a separate JUDGE_OUTPUT_JSONL for every judge model and protocol."
        )


def aggregate(path, mode, model, summary_path):
    rows = read_jsonl(path)
    totals = defaultdict(list)
    usage_totals = defaultdict(int)
    for row in rows:
        for metric, value in row.get("scores", {}).items():
            if value is not None and not (metric == "Temporal Understanding" and float(value) < 0):
                totals[metric].append(float(value))
        for name, value in row.get("token_usage", {}).items():
            if isinstance(value, (int, float)):
                usage_totals[name] += int(value)
    summary = {
        "benchmark": "SVBench",
        "mode": mode,
        "judge_model": model,
        "judge_units": len(rows),
        "token_usage": dict(sorted(usage_totals.items())),
        "scores_0_to_10": {
            metric: (sum(totals[metric]) / len(totals[metric]) if totals[metric] else None)
            for metric in METRICS
        },
    }
    summary["scores_0_to_100"] = {
        metric: (None if value is None else value * 10.0)
        for metric, value in summary["scores_0_to_10"].items()
    }
    expected_units = EXPECTED[mode]["units"]
    summary["average_token_usage_per_unit"] = {
        name: value / max(len(rows), 1) for name, value in sorted(usage_totals.items())
    }
    summary["estimated_full_token_usage"] = {
        name: round(value / max(len(rows), 1) * expected_units)
        for name, value in sorted(usage_totals.items())
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running the SVBench GPT judge.")
    predictions = Path(args.predictions)
    judge_output = Path(args.judge_output)
    summary_output = Path(args.summary_output) if args.summary_output else judge_output.with_suffix(".summary.json")
    judge_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and judge_output.exists():
        judge_output.unlink()
    rows = read_jsonl(predictions)
    groups, sessions = group_predictions(rows, args.mode)
    validate_coverage(rows, groups, sessions, args.mode, args.allow_incomplete)
    validate_existing_judge(judge_output, args.judge_model, args.mode)
    done = completed_units(judge_output)
    pending = [(unit_id, values) for unit_id, values in groups.items() if unit_id not in done]
    if args.limit_units is not None:
        if args.limit_units <= 0:
            raise ValueError("--limit-units must be positive.")
        rng = random.Random(args.sample_seed)
        pending = rng.sample(pending, min(args.limit_units, len(pending)))
    print(json.dumps({"completed_judge_units": len(done), "remaining_judge_units": len(pending)}))

    def evaluate(item):
        unit_id, values = item
        prompt = make_prompt(unit_id, values, args.mode)
        raw, parsed, scores, usage = call_judge(
            api_key,
            args.judge_model,
            prompt,
            args.max_retries,
        )
        return {
            "unit_id": unit_id,
            "mode": args.mode,
            "sample_ids": [row["sample_id"] for row in values],
            "judge_model": args.judge_model,
            "scores": scores,
            "token_usage": usage,
            "prompt_characters": len(prompt),
            "evaluation": parsed,
            "raw_judge_response": raw,
        }

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(evaluate, item): item[0] for item in pending}
        for completed_index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with judge_output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"[svbench-judge] {len(done) + completed_index}/{len(done) + len(pending)} "
                f"{result['unit_id']}",
                flush=True,
            )
    aggregate(judge_output, args.mode, args.judge_model, summary_output)


if __name__ == "__main__":
    main()
