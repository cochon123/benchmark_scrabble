#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [int(item["optimal_score"]) for item in rows]
    placements = [len(item["canonical_optimal_move"]["placements"]) for item in rows]
    words = [
        str(word).upper()
        for item in rows
        for word in item["canonical_optimal_move"].get("words", [])
    ]
    return {
        "games": len({str(item["source_game_id"]) for item in rows}),
        "positions": len(rows),
        "unique_sorted_racks": len({"".join(sorted(str(item["rack"]))) for item in rows}),
        "positions_with_blank_rack": sum("?" in str(item["rack"]) for item in rows),
        "mean_board_tiles": statistics.mean(len(item["board"]) for item in rows),
        "optimal_score": {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "p95": percentile(scores, 0.95),
            "max": max(scores),
        },
        "new_tiles": {
            "mean": statistics.mean(placements),
            "median": statistics.median(placements),
            "seven_tile_moves": sum(count == 7 for count in placements),
        },
        "formed_words": {
            "total": len(words),
            "unique": len(set(words)),
        },
        "by_ply": dict(sorted(Counter(int(item["band_ply"]) for item in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize generated Scrabble splits.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = {
        split: summarize(
            json.loads(
                (args.dataset_dir / f"{split}_positions.json").read_text(encoding="utf-8")
            )
        )
        for split in ("train", "validation", "test")
    }
    output = args.output or (args.dataset_dir / "dataset_statistics.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
