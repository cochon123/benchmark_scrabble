#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any, Callable


VALUES = {
    **dict.fromkeys("AEILNORSTU", 1), **dict.fromkeys("DG", 2),
    **dict.fromkeys("BCMP", 3), **dict.fromkeys("FHVWY", 4),
    "K": 5, **dict.fromkeys("JX", 8), **dict.fromkeys("QZ", 10),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fair_tie_baseline(boards: list[dict[str, Any]], score: Callable[[dict[str, Any]], float]) -> dict[str, Any]:
    rows = []
    for board in boards:
        values = [score(candidate) for candidate in board["candidates"]]
        maximum = max(values)
        tied = [candidate for candidate, value in zip(board["candidates"], values, strict=True) if value == maximum]
        rows.append({
            "id": board["id"], "tie_count": len(tied),
            "optimal_top1_probability": sum(int(candidate["score"]) == int(board["optimal_score"]) for candidate in tied) / len(tied),
            "expected_selected_score": statistics.mean(int(candidate["score"]) for candidate in tied),
            "optimal_score": int(board["optimal_score"]),
        })
    return {
        "optimal_top1_pct": 100 * statistics.mean(row["optimal_top1_probability"] for row in rows),
        "point_ratio_pct": 100 * sum(row["expected_selected_score"] for row in rows) / sum(row["optimal_score"] for row in rows),
        "tie_count_median": statistics.median(row["tie_count"] for row in rows),
        "rows": rows,
    }


def breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "boards": len(group),
            "optimal_top1_pct": 100 * statistics.mean(row["optimal_top1"] for row in group),
            "point_ratio_pct": 100 * sum(row["selected_score"] for row in group) / sum(row["optimal_score"] for row in group),
            "recall_at_32_pct": 100 * statistics.mean(row["recall_at_32"] for row in group),
        }
    by_ply = {str(ply): metrics([row for row in rows if int(row["band_ply"]) == ply]) for ply in range(6, 15)}
    bins = (("le_128", 0, 128), ("129_256", 129, 256), ("257_512", 257, 512), ("gt_512", 513, 10**9))
    by_size = {name: metrics([row for row in rows if low <= int(row["candidate_count"]) <= high]) for name, low, high in bins}
    return {"by_ply": by_ply, "by_candidate_count": by_size}


def bootstrap(primary: list[float], baseline: list[float], seed: int, samples: int = 4000) -> dict[str, float]:
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        indices = [rng.randrange(len(primary)) for _ in primary]
        values.append(statistics.mean(primary[index] - baseline[index] for index in indices))
    values.sort()
    return {"delta": statistics.mean(primary) - statistics.mean(baseline), "lower_95": values[100], "upper_95": values[3899]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    boards = json.loads(args.candidates.read_text(encoding="utf-8"))["boards"]
    primary = json.loads(args.results.read_text(encoding="utf-8"))["primary_details"]
    longest = fair_tie_baseline(boards, lambda candidate: len(candidate["word"]))
    face = fair_tie_baseline(boards, lambda candidate: sum(VALUES[letter] for letter in candidate["word"]))
    primary_top1 = [float(row["optimal_top1"]) for row in primary]
    primary_ratio = [float(row["score_ratio"]) for row in primary]
    payload = {
        "inputs": {"candidates_sha256": sha256(args.candidates), "results_sha256": sha256(args.results)},
        "fair_tie_baselines": {"longest_word": {key: value for key, value in longest.items() if key != "rows"}, "face_value": {key: value for key, value in face.items() if key != "rows"}},
        "fair_tie_bootstrap_comparisons": {
            "top1_vs_longest_word": bootstrap(primary_top1, [row["optimal_top1_probability"] for row in longest["rows"]], 8201),
            "point_ratio_vs_longest_word": bootstrap(primary_ratio, [row["expected_selected_score"] / row["optimal_score"] for row in longest["rows"]], 8202),
            "top1_vs_face_value": bootstrap(primary_top1, [row["optimal_top1_probability"] for row in face["rows"]], 8203),
            "point_ratio_vs_face_value": bootstrap(primary_ratio, [row["expected_selected_score"] / row["optimal_score"] for row in face["rows"]], 8204),
        },
        "primary_breakdown": breakdown(primary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
