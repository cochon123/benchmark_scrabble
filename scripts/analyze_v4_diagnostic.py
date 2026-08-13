#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (100 * (center - margin), 100 * (center + margin))


def binomial_tail(successes: int, total: int, probability: float) -> float:
    return min(
        1.0,
        sum(
            math.comb(total, count) * probability**count * (1 - probability) ** (total - count)
            for count in range(successes, total + 1)
        ),
    )


def mcnemar_exact(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    smaller = min(improved, regressed)
    one_tail = sum(math.comb(discordant, count) for count in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * one_tail)


def enrich(payload: dict[str, Any], random_baselines: dict[str, float] | None = None) -> dict[str, Any]:
    random_baselines = random_baselines or {
        "copy": 0.125,
        "legality": 0.125,
        "anchor_direction": 0.125,
        "ranking": 0.125,
    }
    for label in ("base", "sft"):
        for task, row in payload[label]["by_task"].items():
            successes = int(row["success"])
            total = int(row["boards"])
            low, high = wilson(successes, total)
            row["accuracy_wilson_95_pct"] = [low, high]
            probability = float(random_baselines[task])
            row["random_choice_baseline_pct"] = 100 * probability
            row["p_value_above_random"] = binomial_tail(successes, total, probability)
    sft = payload["sft"]["by_task"]
    copy = float(sft["copy"]["accuracy_pct"])
    legality = float(sft["legality"]["accuracy_pct"])
    ranking = float(sft["ranking"]["accuracy_pct"])
    anchor = float(sft["anchor_direction"]["accuracy_pct"])
    ranking_legal = float(sft["ranking"]["legal_output_pct"])
    if ranking >= 35 and ranking_legal >= 75:
        diagnosis = "free_search_failure_with_candidate_skill"
        action = "Use candidate and anchor scaffolding as the next curriculum, progressively remove it, and postpone RL until unaided legality clears 10%."
    elif copy < 80:
        diagnosis = "execution_and_coordinate_copying_failure"
        action = "Train deterministic candidate copying and coordinate grounding before any Scrabble reasoning objective."
    elif legality < 40 or anchor < 40:
        diagnosis = "board_legality_and_anchor_failure"
        action = "Train anchor/direction and verifier-error curricula with one skill per example; do not use RL yet."
    elif ranking < 35:
        diagnosis = "score_ranking_failure_after_legality"
        action = "Keep constrained legal candidates and train score-aware ranking before removing scaffolds."
    else:
        diagnosis = "mixed_skill_failure"
        action = "Expand the decomposed evaluation before another paid training run."
    payload["diagnosis"] = {
        "classification": diagnosis,
        "recommended_action": action,
        "thresholds_pct": {"copy": 80, "legality": 40, "anchor_direction": 40, "ranking": 35},
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/v4_diagnostic/analysis.json"))
    parser.add_argument("--dataset", type=Path, default=Path("data/general_v4_diagnostic/diagnostic.json"))
    args = parser.parse_args()
    records = json.loads(args.dataset.read_text(encoding="utf-8"))
    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(str(record["task"]), []).append(record)
    baselines = {}
    for task, rows in by_task.items():
        acceptable = 0
        candidates = 0
        for row in rows:
            candidates += len(row["candidates"])
            if task == "ranking":
                acceptable += sum(
                    bool(candidate["legal"])
                    and int(candidate["score"]) == int(row["position"]["optimal_score"])
                    for candidate in row["candidates"]
                )
            else:
                acceptable += 1
        baselines[task] = acceptable / candidates
    payload = enrich(json.loads(args.comparison.read_text(encoding="utf-8")), baselines)
    payload["protocol"]["exact_random_choice_baseline_pct"] = {
        task: 100 * probability for task, probability in sorted(baselines.items())
    }
    base_path = args.comparison.with_name("base.json")
    sft_path = args.comparison.with_name("sft.json")
    if base_path.exists() and sft_path.exists():
        base_results = {row["id"]: row for row in json.loads(base_path.read_text(encoding="utf-8"))["results"]}
        sft_results = {row["id"]: row for row in json.loads(sft_path.read_text(encoding="utf-8"))["results"]}
        paired = {}
        for task in sorted(by_task):
            ids = [row["id"] for row in records if row["task"] == task]
            improved = sum(not base_results[item]["success"] and sft_results[item]["success"] for item in ids)
            regressed = sum(base_results[item]["success"] and not sft_results[item]["success"] for item in ids)
            paired[task] = {
                "improved": improved,
                "regressed": regressed,
                "both_succeeded": sum(base_results[item]["success"] and sft_results[item]["success"] for item in ids),
                "both_failed": sum(not base_results[item]["success"] and not sft_results[item]["success"] for item in ids),
                "mcnemar_exact_two_sided_p": mcnemar_exact(improved, regressed),
            }
        payload["paired_base_vs_sft"] = paired
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
