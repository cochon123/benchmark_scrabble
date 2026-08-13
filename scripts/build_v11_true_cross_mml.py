#!/usr/bin/env python3
"""Build verifier-certified true-cross multi-positive V11 data."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.constants import TILE_DISTRIBUTION
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.solver import (
    BoardTile,
    empty_grid,
    enumerate_moves,
    grid_from_position,
    grid_to_cells,
    infer_orientation,
    read_word,
    validate_and_score_move,
    word_text,
)
from scrabble_bench.v11_training import gate_row, move_action, policy_messages


ROOT = Path(__file__).resolve().parents[1]
K_POSITIVES = 8


def tile_bag() -> list[str]:
    return [letter for letter, count in TILE_DISTRIBUTION.items() for _ in range(count)]


def opening_successor(*, index: int, seed: int, lexicon: Lexicon, rng: random.Random) -> dict[str, Any] | None:
    bag = tile_bag()
    rng.shuffle(bag)
    rack1 = "".join(sorted(bag.pop() for _ in range(7)))
    openings = enumerate_moves(lexicon, empty_grid(), rack1)
    if not openings:
        return None
    opening = rng.choice(openings[: min(40, len(openings))])
    board = empty_grid()
    for placement in opening.placements:
        board[placement.row][placement.col] = BoardTile(placement.letter, placement.is_blank)
    rack2 = "".join(sorted(bag.pop() for _ in range(7)))
    return {
        "id": f"v11-true-cross-{seed}-{index:04d}",
        "source_game_id": f"v11-true-cross-{seed}-{index:04d}",
        "band_ply": 1,
        "board": grid_to_cells(board),
        "rack": rack2,
    }


def move_features(position: dict[str, Any], move: Any, lexicon: Lexicon) -> dict[str, Any] | None:
    placements = [item.to_dict() for item in move.placements]
    try:
        validated = validate_and_score_move(
            lexicon, grid_from_position(position["board"]), str(position["rack"]), placements
        )
    except ValueError:
        return None
    grid = grid_from_position(position["board"])
    for placement in validated.placements:
        grid[placement.row][placement.col] = BoardTile(placement.letter, placement.is_blank)
    orientation = infer_orientation(validated.placements)
    if orientation == "across":
        main = read_word(grid, validated.placements[0].row, validated.placements[0].col, 0, 1)
        cross_direction = (1, 0)
    elif orientation == "down":
        main = read_word(grid, validated.placements[0].row, validated.placements[0].col, 1, 0)
        cross_direction = (0, 1)
    else:
        main = []
        cross_direction = (0, 1)
    cross_words = []
    for placement in validated.placements:
        cross = read_word(grid, placement.row, placement.col, *cross_direction)
        if len(cross) > 1:
            cross_words.append(word_text(cross))
    if not cross_words:
        return None
    main_word = word_text(main) if main else (validated.words[0] if validated.words else "")
    action = move_action(position, validated, lexicon)
    row, col = validated.placements[0].row, validated.placements[0].col
    return {
        "action": action,
        "direction": orientation or "single",
        "main_word": main_word,
        "word_length": len(main_word),
        "new_tiles": len(validated.placements),
        "cross_count": len(cross_words),
        "cross_words": sorted(set(cross_words)),
        "region": f"{row // 5},{col // 5}",
        "score": int(validated.score),
    }


def select_diverse(candidates: list[dict[str, Any]], *, count: int = K_POSITIVES) -> list[dict[str, Any]]:
    if len(candidates) < count:
        return []
    remaining = sorted(candidates, key=lambda item: (-item["score"], item["action"]))
    chosen: list[dict[str, Any]] = []
    seen: dict[str, set[Any]] = {key: set() for key in ("direction", "word_length", "new_tiles", "cross_count", "region")}
    while remaining and len(chosen) < count:
        best = max(
            remaining,
            key=lambda item: (
                sum(item[key] not in seen[key] for key in seen),
                item["cross_count"],
                item["new_tiles"],
                item["word_length"],
                item["score"],
            ),
        )
        chosen.append(best)
        remaining.remove(best)
        for key in seen:
            seen[key].add(best[key])
    return chosen


def materialize(count: int, seed: int, lexicon: Lexicon) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError(f"Could not make {count} true-cross positions; kept {len(rows)}")
        position = opening_successor(index=len(rows), seed=seed, lexicon=lexicon, rng=rng)
        if position is None:
            continue
        legal = enumerate_moves(lexicon, grid_from_position(position["board"]), position["rack"])
        candidates = [feature for move in legal if (feature := move_features(position, move, lexicon))]
        selected = select_diverse(candidates)
        if len(selected) != K_POSITIVES:
            continue
        position["true_cross_metadata"] = {
            "candidate_count": len(candidates),
            "selected": selected,
            "all_selected_have_perpendicular_new_word": True,
        }
        rows.append(position)
        if len(rows) % 50 == 0:
            print(f"built {len(rows)}/{count}", flush=True)
    return rows


def mml_record(position: dict[str, Any]) -> dict[str, Any]:
    selected = position["true_cross_metadata"]["selected"]
    return {
        "id": f"{position['id']}--v11-true-cross-mml",
        "source_id": position["id"],
        "source_game_id": position["source_game_id"],
        "record_type": "v11-true-cross-mml",
        "position_key": position["id"],
        "messages": policy_messages(position),
        "positive_actions": [item["action"] for item in selected],
        "positive_metadata": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--validation-count", type=int, default=80)
    parser.add_argument("--gate-count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=13001)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/v11_true_cross_mml")
    args = parser.parse_args()
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    train_positions = materialize(args.train_count, args.seed, lexicon)
    validation_positions = materialize(args.validation_count, args.seed + 1, lexicon)
    gate_positions = materialize(args.gate_count, args.seed + 2, lexicon)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.jsonl").write_text(
        "".join(json.dumps(mml_record(item), separators=(",", ":"), ensure_ascii=False) + "\n" for item in train_positions),
        encoding="utf-8",
    )
    (args.output_dir / "validation.jsonl").write_text(
        "".join(json.dumps(mml_record(item), separators=(",", ":"), ensure_ascii=False) + "\n" for item in validation_positions),
        encoding="utf-8",
    )
    gate_rows = [gate_row(item, category="true_cross") for item in gate_positions]
    (args.output_dir / "gate.json").write_text(json.dumps(gate_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "train": len(train_positions),
        "validation": len(validation_positions),
        "gate": len(gate_positions),
        "positives_per_prompt": K_POSITIVES,
        "seed": args.seed,
        "train_ids": [item["id"] for item in train_positions],
        "gate_ids": [item["id"] for item in gate_positions],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("train", "validation", "gate", "positives_per_prompt", "seed")}))


if __name__ == "__main__":
    main()
