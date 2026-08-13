from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .lexicon import Lexicon
from .v4_diagnostic import clean_placements
from .v41_training import move_plan
from .v7_training import plan_to_placements


PLAN_COMPONENTS = ("row", "col", "direction", "word")


def _plan_key(plan: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        int(plan["row"]),
        int(plan["col"]),
        str(plan["direction"]),
        str(plan["word"]).upper(),
    )


def plan_from_move(position: dict[str, Any], move: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    plan = move_plan(position, clean_placements(move["placements"]), lexicon)
    return {
        "row": int(plan["start_row"]),
        "col": int(plan["start_col"]),
        "direction": str(plan["direction"]),
        "word": str(plan["word"]).upper(),
        "score": int(move["score"]),
    }


def stored_moves(position: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        position.get("canonical_optimal_move"),
        *(position.get("optimal_moves") or []),
        *(position.get("candidate_moves") or []),
    ]
    unique: dict[str, dict[str, Any]] = {}
    for move in candidates:
        if not move or "placements" not in move:
            continue
        key = json.dumps(clean_placements(move["placements"]), sort_keys=True)
        unique.setdefault(key, move)
    return list(unique.values())


def accepted_plans(
    position: dict[str, Any],
    lexicon: Lexicon,
    *,
    minimum_score_ratio: float = 1.0,
) -> list[dict[str, Any]]:
    threshold = float(position["optimal_score"]) * minimum_score_ratio
    plans: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for move in stored_moves(position):
        if int(move.get("score", 0)) + 1e-9 < threshold:
            continue
        try:
            plan = plan_from_move(position, move, lexicon)
        except (RuntimeError, ValueError):
            continue
        plans.setdefault(_plan_key(plan), plan)
    return sorted(plans.values(), key=lambda item: (-int(item["score"]), _plan_key(item)))


def component_acceptance(plans: Iterable[dict[str, Any]]) -> dict[str, set[Any]]:
    plan_list = list(plans)
    return {name: {plan[name] for plan in plan_list} for name in PLAN_COMPONENTS}


def independent_component_matches(
    predicted: dict[str, Any], plans: Iterable[dict[str, Any]]
) -> dict[str, bool]:
    accepted = component_acceptance(plans)
    return {
        name: predicted.get(name) in accepted[name]
        for name in PLAN_COMPONENTS
        if name in predicted
    }


def best_constant_baselines(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    row_list = list(rows)
    output: dict[str, dict[str, Any]] = {}
    for component in PLAN_COMPONENTS:
        counts = Counter(row["target_plan"][component] for row in row_list)
        value, correct = counts.most_common(1)[0]
        output[component] = {
            "value": value,
            "correct": correct,
            "total": len(row_list),
            "accuracy_pct": 100 * correct / len(row_list),
        }
    locations = Counter(
        (int(row["target_plan"]["row"]), int(row["target_plan"]["col"]))
        for row in row_list
    )
    value, correct = locations.most_common(1)[0]
    output["location"] = {
        "value": list(value),
        "correct": correct,
        "total": len(row_list),
        "accuracy_pct": 100 * correct / len(row_list),
    }
    return output


@dataclass(frozen=True)
class RepairResult:
    legal: bool
    score: int
    plan: dict[str, Any] | None


def verify_plan(position: dict[str, Any], plan: dict[str, Any], lexicon: Lexicon) -> RepairResult:
    try:
        _, score = plan_to_placements(position, plan, lexicon)
    except (RuntimeError, ValueError):
        return RepairResult(False, 0, None)
    return RepairResult(True, int(score), dict(plan))


def best_component_repair(
    position: dict[str, Any],
    predicted: dict[str, Any],
    accepted: Iterable[dict[str, Any]],
    components: Iterable[str],
    lexicon: Lexicon,
) -> RepairResult:
    names = tuple(components)
    best = RepairResult(False, 0, None)
    for target in accepted:
        candidate = dict(predicted)
        for name in names:
            candidate[name] = target[name]
        result = verify_plan(position, candidate, lexicon)
        if result.legal and (not best.legal or result.score > best.score):
            best = result
    return best

