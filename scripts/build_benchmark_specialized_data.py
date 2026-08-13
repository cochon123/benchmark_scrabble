#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scrabble_bench.config import DATASET_PATH
from scrabble_bench.training import TRANSFORMS, training_record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an explicitly benchmark-contaminated SFT set. "
            "Use only to measure memorization/specialization, never held-out ability."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training/benchmark_specialized_train.jsonl"),
    )
    parser.add_argument("--boards", type=int, default=None)
    args = parser.parse_args()

    positions = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.boards is not None:
        positions = positions[: args.boards]
    records = [
        {
            **training_record(position, transform),
            "benchmark_contaminated": True,
            "warning": "This record is from the evaluation benchmark and cannot support held-out claims.",
        }
        for position in positions
        for transform in TRANSFORMS
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "source": str(args.dataset),
        "boards": len(positions),
        "records": len(records),
        "transforms": list(TRANSFORMS),
        "benchmark_contaminated": True,
        "valid_for_held_out_claims": False,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
