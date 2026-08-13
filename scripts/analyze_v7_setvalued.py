#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.v8_audit import (
    PLAN_COMPONENTS,
    accepted_plans,
    best_component_repair,
    best_constant_baselines,
    independent_component_matches,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-optimal-ratio", type=float, default=0.8)
    args = parser.parse_args()

    lexicon_path = resolve_lexicon_path()
    lexicon = Lexicon.from_path(lexicon_path)
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in rows}
    results = evaluation["results"]

    component_counts = Counter()
    component_totals = Counter()
    canonical_counts = Counter()
    prediction_modes: dict[str, Counter[Any]] = {name: Counter() for name in PLAN_COMPONENTS}
    repair_specs = {
        "row": ("row",),
        "column": ("col",),
        "location": ("row", "col"),
        "direction": ("direction",),
        "word": ("word",),
        "slot": ("row", "col", "direction"),
        "full_plan": PLAN_COMPONENTS,
    }
    repairs = {name: Counter() for name in repair_specs}
    multiplicity = Counter()
    lexicon_words = 0
    detailed: list[dict[str, Any]] = []

    for result in results:
        row = by_id[result["id"]]
        position = row["position"]
        predicted = result.get("predicted") or {}
        optimal = accepted_plans(position, lexicon, minimum_score_ratio=1.0)
        near = accepted_plans(
            position, lexicon, minimum_score_ratio=args.near_optimal_ratio
        )
        multiplicity[len(optimal)] += 1
        matches = independent_component_matches(predicted, optimal)
        canonical = {
            name: predicted.get(name) == row["target_plan"][name]
            for name in PLAN_COMPONENTS
            if name in predicted
        }
        for name, correct in matches.items():
            component_counts[name] += bool(correct)
            component_totals[name] += 1
            canonical_counts[name] += bool(canonical[name])
            prediction_modes[name][predicted[name]] += 1
        if str(predicted.get("word", "")) in lexicon.words:
            lexicon_words += 1

        repair_detail = {}
        if all(name in predicted for name in PLAN_COMPONENTS):
            for label, components in repair_specs.items():
                repaired = best_component_repair(
                    position, predicted, near, components, lexicon
                )
                repairs[label]["legal"] += repaired.legal
                repairs[label]["optimal"] += (
                    repaired.legal and repaired.score == int(position["optimal_score"])
                )
                repairs[label]["score"] += repaired.score
                repairs[label]["boards"] += 1
                repair_detail[label] = {
                    "legal": repaired.legal,
                    "score": repaired.score,
                    "plan": repaired.plan,
                }
        detailed.append(
            {
                "id": result["id"],
                "source_game_id": row["source_game_id"],
                "predicted": predicted,
                "ordered_first_error": result.get("first_error"),
                "independent_optimal_component_match": matches,
                "canonical_component_match": canonical,
                "optimal_plan_count": len(optimal),
                "near_optimal_plan_count": len(near),
                "repairs": repair_detail,
            }
        )

    free_rows = [row for row in rows if row["task"] == "free-plan"]
    summary: dict[str, Any] = {
        "version": "v8-setvalued-audit-1",
        "records": len(results),
        "lexicon": {
            "path": str(lexicon_path),
            "sha256": hashlib.sha256(lexicon_path.read_bytes()).hexdigest(),
            "words": len(lexicon.words),
            "predicted_words_in_lexicon": lexicon_words,
        },
        "ordered_first_errors": dict(
            sorted(Counter(str(row.get("first_error")) for row in results).items())
        ),
        "independent_optimal_component_accuracy_pct": {
            name: 100 * component_counts[name] / component_totals[name]
            for name in component_totals
        },
        "canonical_component_accuracy_pct": {
            name: 100 * canonical_counts[name] / component_totals[name]
            for name in component_totals
        },
        "prediction_modes": {
            name: [[value, count] for value, count in counts.most_common(10)]
            for name, counts in prediction_modes.items()
        },
        "optimal_plan_multiplicity": dict(sorted(multiplicity.items())),
        "constant_baselines": best_constant_baselines(free_rows),
        "counterfactual_repairs": {
            name: {
                "boards": values["boards"],
                "legal": values["legal"],
                "legal_pct": 100 * values["legal"] / values["boards"],
                "optimal": values["optimal"],
                "optimal_pct": 100 * values["optimal"] / values["boards"],
                "score_pct": 100
                * values["score"]
                / sum(by_id[row["id"]]["position"]["optimal_score"] for row in results),
            }
            for name, values in repairs.items()
            if values["boards"]
        },
        "hashes": {
            "dataset": sha256(args.dataset),
            "evaluation": sha256(args.evaluation),
        },
    }
    payload = {"summary": summary, "results": detailed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
