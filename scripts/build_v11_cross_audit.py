#!/usr/bin/env python3
"""Freeze a fresh, lineage-disjoint controlled-cross evaluation set for V11."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.v11_training import gate_row
from scripts.build_v11_curriculum import synthetic_controlled_gate


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12011)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/v11_atomic_policy/cross_identifiability_gate.json",
    )
    args = parser.parse_args()
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    positions = synthetic_controlled_gate(count=args.count, seed=args.seed, lexicon=lexicon)
    rows = [gate_row(position, category="controlled_cross") for position in positions]
    source_ids = {row["source_id"] for row in rows}
    if len(rows) != args.count or len(source_ids) != len(rows):
        raise RuntimeError("Fresh cross audit must contain exactly one unique lineage per row.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "positions": len(rows), "seed": args.seed}))


if __name__ == "__main__":
    main()
