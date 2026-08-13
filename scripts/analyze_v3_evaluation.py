#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scrabble_bench.evaluation import summarize_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report legality, recovery, score-claim, and attempt metrics."
    )
    parser.add_argument("evaluation", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    reports = []
    for path in args.evaluation:
        payload = json.loads(path.read_text(encoding="utf-8"))
        reports.append(
            {
                "path": str(path),
                "model_summary": payload.get("summary", {}),
                "legality_report": summarize_evaluation(payload),
            }
        )
    output = {"evaluations": reports}
    rendered = json.dumps(output, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
