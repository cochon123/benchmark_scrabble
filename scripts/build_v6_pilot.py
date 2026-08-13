#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from scrabble_bench.config import DATASET_PATH, resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.runner import prompt_for_position
from scrabble_bench.training import position_key
from scrabble_bench.v4_diagnostic import clean_placements
from scrabble_bench.v6_training import board_encoding, direct_record, search_record, value_record


ROOT = Path(__file__).resolve().parents[1]
SEED = 8601


def canonical_hash(rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def gate_ids() -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    games: set[str] = set()
    for relative in (
        "data/general_v42/localization_gate.json",
        "data/general_v41/plan_gate.json",
        "data/general_v42_passk/positions.json",
    ):
        for row in json.loads((ROOT / relative).read_text(encoding="utf-8")):
            ids.add(str(row["id"]))
            games.add(str(row.get("source_game_id", "")))
    return ids, games


def build_rows(
    positions: list[dict[str, Any]],
    builder: Callable[..., dict[str, Any]],
    lexicon: Lexicon,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, position in enumerate(positions):
        first = "transpose" if index % 2 else "identity"
        second = "identity" if first == "transpose" else "transpose"
        if builder is value_record:
            rows.append(direct_record(position, lexicon, transform=first, seed=seed + index))
            rows.append(builder(position, lexicon, transform=second, seed=seed + index))
        else:
            rows.append(builder(position, lexicon, transform=first, seed=seed + index))
            rows.append(builder(position, lexicon, transform=second, seed=seed + index + 1))
    random.Random(seed).shuffle(rows)
    return rows


def selection_set(positions: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    by_game: dict[str, list[dict[str, Any]]] = {}
    for position in positions:
        by_game.setdefault(str(position["source_game_id"]), []).append(position)
    selected: list[dict[str, Any]] = []
    for game in sorted(by_game)[:limit]:
        position = min(by_game[game], key=lambda row: (abs(int(row["band_ply"]) - 10), row["id"]))
        messages = prompt_for_position(
            position,
            board_encoding=board_encoding(str(position["id"]), SEED + 200_000, dense_rate=0.5),
        )
        selected.append(
            {
                "id": f"{position['id']}--v6-selection-free",
                "source_id": position["id"],
                "source_game_id": position["source_game_id"],
                "band_ply": int(position["band_ply"]),
                "messages": messages,
                "position": position,
                "target_placements": clean_placements(
                    position["canonical_optimal_move"]["placements"]
                ),
            }
        )
    if len(selected) != limit:
        raise RuntimeError(f"Need {limit} selection games, found {len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions-dir", type=Path, default=ROOT / "data/general_v6_positions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/general_v6_pilot")
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--validation-limit", type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    train = json.loads((args.positions_dir / "train_positions.json").read_text(encoding="utf-8"))
    validation = json.loads((args.positions_dir / "validation_positions.json").read_text(encoding="utf-8"))
    train = train[: args.train_limit]
    validation = validation[: args.validation_limit]

    official_keys = {position_key(row) for row in json.loads(DATASET_PATH.read_text(encoding="utf-8"))}
    frozen_ids, frozen_games = gate_ids()
    train_ids = {str(row["id"]) for row in train}
    train_games = {str(row["source_game_id"]) for row in train}
    validation_games = {str(row["source_game_id"]) for row in validation}
    if train_ids & frozen_ids or train_games & frozen_games:
        raise RuntimeError("V6 train/frozen gate overlap")
    if train_games & validation_games:
        raise RuntimeError("V6 whole-game train/validation overlap")
    if any(position_key(row) in official_keys for row in train + validation):
        raise RuntimeError("Official benchmark overlap")

    builders = {
        "direct": direct_record,
        "action_value": value_record,
        "compact_search": search_record,
    }
    hashes: dict[str, str] = {}
    stats: dict[str, Any] = {}
    for name, builder in builders.items():
        rows = build_rows(train, builder, lexicon, SEED)
        validation_rows = build_rows(validation, builder, lexicon, SEED + 100_000)
        hashes[f"{name}_train"] = write_jsonl(args.output_dir / f"{name}_train.jsonl", rows)
        hashes[f"{name}_validation"] = write_jsonl(
            args.output_dir / f"{name}_validation.jsonl", validation_rows
        )
        stats[name] = {
            "train_records": len(rows),
            "validation_records": len(validation_rows),
            "types": dict(sorted(Counter(row["record_type"] for row in rows).items())),
            "canonical_sha256": canonical_hash(rows),
        }

    selection = selection_set(validation)
    selection_payload = json.dumps(selection, separators=(",", ":"))
    (args.output_dir / "selection_passk.json").write_text(selection_payload, encoding="utf-8")
    hashes["selection_passk"] = hashlib.sha256(selection_payload.encode()).hexdigest()

    manifest = {
        "version": "v6-4b-action-value-pilot-1",
        "seed": SEED,
        "train_positions": len(train),
        "validation_positions": len(validation),
        "distinct_train_games": len(train_games),
        "candidate_strategy": "top-4-plus-score-quantiles",
        "official_positions_used": 0,
        "train_validation_game_overlap": 0,
        "train_frozen_gate_overlap": 0,
        "selection_boards": len(selection),
        "variants": stats,
        "hashes": hashes,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
