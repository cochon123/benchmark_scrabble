#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail unless an evaluation beats the completed closed-model smoke target."
    )
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--target-pct", type=float, default=52.53164556962025)
    parser.add_argument("--expected-boards", type=int, default=5)
    parser.add_argument("--expected-optimal-points", type=int, default=158)
    args = parser.parse_args()

    payload = json.loads(args.evaluation.read_text(encoding="utf-8"))
    summary = payload["summary"]
    checks = {
        "board_count_matches": int(summary["boards"]) == args.expected_boards,
        "denominator_matches": int(summary["optimal_points"]) == args.expected_optimal_points,
        "beats_target": float(summary["score_pct"]) > args.target_pct,
    }
    result = {
        "evaluation": str(args.evaluation),
        "score_pct": float(summary["score_pct"]),
        "target_pct": args.target_pct,
        "margin_points": float(summary["score_pct"]) - args.target_pct,
        "checks": checks,
        "passed": all(checks.values()),
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
