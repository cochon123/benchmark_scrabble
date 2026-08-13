#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scrabble_bench.config import DATASET_PATH, resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.solver import grid_from_position, validate_and_score_move
from scrabble_bench.training import position_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a generated corpus for leakage, split isolation, and label validity."
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    official = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    official_keys = {position_key(item) for item in official}
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_keys: dict[str, set[str]] = {}
    split_games: dict[str, set[str]] = {}
    label_errors: list[dict[str, Any]] = []
    official_overlap: list[str] = []

    for split in ("train", "validation", "test"):
        path = args.dataset_dir / f"{split}_positions.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        split_rows[split] = rows
        split_keys[split] = {position_key(item) for item in rows}
        split_games[split] = {str(item["source_game_id"]) for item in rows}
        for item in rows:
            key = position_key(item)
            if key in official_keys:
                official_overlap.append(str(item["id"]))
            try:
                move = validate_and_score_move(
                    lexicon,
                    grid_from_position(item["board"]),
                    item["rack"],
                    item["canonical_optimal_move"]["placements"],
                )
                if move.score != int(item["optimal_score"]):
                    raise ValueError(
                        f"label score {item['optimal_score']} != validated score {move.score}"
                    )
            except Exception as exc:
                label_errors.append({"id": item["id"], "error": str(exc)})

    cross_split_positions: dict[str, int] = {}
    cross_split_games: dict[str, int] = {}
    splits = tuple(split_rows)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            label = f"{left}__{right}"
            cross_split_positions[label] = len(split_keys[left] & split_keys[right])
            cross_split_games[label] = len(split_games[left] & split_games[right])

    selection_path = args.dataset_dir / "selection_positions.json"
    selection = (
        json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_path.exists()
        else []
    )
    selection_keys = {position_key(item) for item in selection}
    selection_outside_validation = selection_keys - split_keys["validation"]
    selection_train_overlap = selection_keys & split_keys["train"]
    selection_test_overlap = selection_keys & split_keys["test"]

    duplicate_ids: dict[str, list[str]] = {}
    all_ids: defaultdict[str, list[str]] = defaultdict(list)
    for split, rows in split_rows.items():
        for row in rows:
            all_ids[str(row["id"])].append(split)
    duplicate_ids = {
        row_id: owners for row_id, owners in all_ids.items() if len(owners) != 1
    }

    summary: dict[str, Any] = {
        "dataset_dir": str(args.dataset_dir),
        "official_reference_positions": len(official),
        "official_overlap_count": len(official_overlap),
        "official_overlap_ids": official_overlap,
        "cross_split_position_duplicates": cross_split_positions,
        "cross_split_game_duplicates": cross_split_games,
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids,
        "invalid_label_count": len(label_errors),
        "invalid_labels": label_errors,
        "selection": {
            "positions": len(selection),
            "outside_validation": len(selection_outside_validation),
            "train_overlap": len(selection_train_overlap),
            "test_overlap": len(selection_test_overlap),
            "by_ply": dict(
                sorted(Counter(int(item["band_ply"]) for item in selection).items())
            ),
        },
        "splits": {
            split: {
                "games": len(split_games[split]),
                "positions": len(rows),
                "unique_position_keys": len(split_keys[split]),
                "by_ply": dict(
                    sorted(Counter(int(item["band_ply"]) for item in rows).items())
                ),
                "mean_optimal_score": (
                    sum(int(item["optimal_score"]) for item in rows) / len(rows)
                    if rows
                    else 0
                ),
            }
            for split, rows in split_rows.items()
        },
    }
    passed = (
        not official_overlap
        and not any(cross_split_positions.values())
        and not any(cross_split_games.values())
        and not duplicate_ids
        and not label_errors
        and not selection_outside_validation
        and not selection_train_overlap
        and not selection_test_overlap
    )
    summary["passed"] = passed
    output = args.output or (args.dataset_dir / "audit.json")
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit("Dataset audit failed")


if __name__ == "__main__":
    main()
