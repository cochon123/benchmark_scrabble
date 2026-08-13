#!/usr/bin/env python3
"""Materialize a randomized single-positive control from the true-cross set."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def convert(source: Path, destination: Path, seed: int) -> None:
    rng = random.Random(seed)
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        actions = list(record["positive_actions"])
        action_index = rng.randrange(len(actions))
        rows.append(
            {
                "id": f"{record['id']}--randomized-sft",
                "source_id": record["source_id"],
                "source_game_id": record["source_game_id"],
                "record_type": "v11-true-cross-randomized-sft",
                "messages": record["messages"] + [{"role": "assistant", "content": actions[action_index]}],
                "selected_positive_index": action_index,
                "positive_count": len(actions),
            }
        )
    destination.write_text(
        "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/v11_true_cross_mml"))
    parser.add_argument("--seed", type=int, default=13021)
    args = parser.parse_args()
    convert(args.input_dir / "train.jsonl", args.input_dir / "train_sft.jsonl", args.seed)
    convert(args.input_dir / "validation.jsonl", args.input_dir / "validation_sft.jsonl", args.seed + 1)
    print(json.dumps({"train": str(args.input_dir / "train_sft.jsonl"), "validation": str(args.input_dir / "validation_sft.jsonl"), "seed": args.seed}))


if __name__ == "__main__":
    main()
