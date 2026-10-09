# Adapted from EvolvingLMMs-Lab/lmms-eval v0.3.5 (Apache-2.0).

import ast
import os
import random
import re
from collections import defaultdict

from loguru import logger as eval_logger


def longvideobench_candidates(doc):
    """Read the official `candidates` field, with lmms-eval schema fallback."""
    candidates = doc.get("candidates")
    if candidates is not None:
        if isinstance(candidates, str):
            candidates = ast.literal_eval(candidates)
        return list(candidates)
    result = []
    for i in range(5):
        candidate = doc.get(f"option{i}")
        if candidate in {None, "N/A"}:
            break
        result.append(candidate)
    return result


def longvideobench_doc_to_text(doc, lmms_eval_specific_kwargs):
    candidates = longvideobench_candidates(doc)
    question = "Question: " + doc["question"] + "\n" + "\n".join(
        ". ".join([chr(ord("A") + i), candidate])
        for i, candidate in enumerate(candidates)
    )
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    return f"{pre_prompt}{question}\n{post_prompt}"


def longvideobench_doc_to_visual_v(doc):
    cache_root = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
    return [os.path.join(cache_root, "longvideobench", "videos", doc["video_path"])]


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Changed from MMMU-style complex parsing into simple parsing.
    Fixed to avoid 'D. A book' be parsed as A.
    Same as original LongVideoBench paper (from author Haoning Wu), if parsing failed, it will assign a random choice to model.
    """
    s = response.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return random.choice(all_choices)

    matches = re.search(r"[ABCDE]", s)
    if matches is None:
        return random.choice(all_choices)
    return matches[0]


def evaluate_longvideobench(samples):
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        correct = eval_multi_choice(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return judge_dict, {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def eval_multi_choice(gold_i, pred_i):
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def longvideobench_process_results(doc, results):
    pred = results[0]
    all_choices = []
    index2ans = {}
    for i, option in enumerate(longvideobench_candidates(doc)):
        index2ans[chr(ord("A") + i)] = option
        all_choices.append(chr(ord("A") + i))

    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    id = doc["id"]
    lvb_acc = {"id": id, "duration_group": doc["duration_group"], "question_category": doc["question_category"], "answer": chr(ord("A") + doc["correct_choice"]), "parsed_pred": parsed_pred}
    return {
        "lvb_acc": lvb_acc,
        "submission": {
            id: pred,
        },
    }


def longvideobench_aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["duration_group"]].append(result)
        subset_to_eval_samples[result["question_category"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_longvideobench(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}

    for cat_name, cat_results in evaluation_result.items():
        printable_results[cat_name] = {
            "num": int(cat_results["num_example"]),
            "acc": round(cat_results["acc"], 5),
        }
    _, overall_metric = evaluate_longvideobench(results)
    printable_results["Overall"] = {
        "num": len(results),
        "acc": round(overall_metric["acc"], 5),
    }
    eval_logger.info(printable_results)
    return printable_results["Overall"]["acc"]
