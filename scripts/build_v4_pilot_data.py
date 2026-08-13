#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scrabble_bench.config import DATASET_PATH, resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.training import position_key
from scrabble_bench.v4_training import (
    full_move_record,
    preference_record,
    ranking_record,
    recovery_record,
)


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def select_train_positions(
    positions: list[dict[str, Any]],
    *,
    count: int,
    opening_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    openings = [item for item in positions if int(item["band_ply"]) == 0]
    non_openings = [item for item in positions if int(item["band_ply"]) > 0]
    openings.sort(key=lambda item: stable_key(seed, str(item["id"])))
    non_openings.sort(key=lambda item: stable_key(seed, str(item["id"])))
    opening_count = min(round(count * opening_fraction), len(openings))
    selected = openings[:opening_count] + non_openings[: count - opening_count]
    selected.sort(key=lambda item: stable_key(seed + 1, str(item["id"])))
    if len(selected) != count:
        raise RuntimeError(f"Requested {count} train positions but selected {len(selected)}")
    return selected


def split_validation_games(
    positions: list[dict[str, Any]],
    *,
    gate_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position in positions:
        by_game[str(position["source_game_id"])].append(position)
    games = sorted(by_game, key=lambda value: stable_key(seed, value))
    if len(games) <= gate_count:
        raise RuntimeError("Not enough validation games to isolate loss and gate sets")
    gate_games = games[:gate_count]
    loss_games = set(games[gate_count:])
    gate: list[dict[str, Any]] = []
    target_plies = list(range(1, 15))
    for index, game in enumerate(gate_games):
        target = target_plies[index % len(target_plies)]
        choices = sorted(by_game[game], key=lambda item: abs(int(item["band_ply"]) - target))
        selected = next(item for item in choices if int(item["band_ply"]) > 0)
        gate.append(selected)
    loss = [item for item in positions if str(item["source_game_id"]) in loss_games]
    return gate, loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the short, verifier-derived v4 pilot corpus.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/general_v3_merged"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/general_v4_pilot"))
    parser.add_argument("--train-positions", type=int, default=4000)
    parser.add_argument("--opening-fraction", type=float, default=0.05)
    parser.add_argument("--gate-positions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4407)
    args = parser.parse_args()

    train_source = read_json(args.source_dir / "train_positions.json")
    validation_source = read_json(args.source_dir / "validation_positions.json")
    official = read_json(DATASET_PATH)
    official_keys = {position_key(item) for item in official}
    lexicon = Lexicon.from_path(resolve_lexicon_path())

    selected = select_train_positions(
        train_source,
        count=args.train_positions,
        opening_fraction=args.opening_fraction,
        seed=args.seed,
    )
    gate, loss_positions = split_validation_games(
        validation_source,
        gate_count=args.gate_positions,
        seed=args.seed,
    )
    if any(position_key(item) in official_keys for item in selected + gate + loss_positions):
        raise RuntimeError("Official benchmark overlap detected")

    train_rows: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    for index, position in enumerate(selected):
        train_rows.append(full_move_record(position))
        if index % 2 == 0:
            train_rows.append(ranking_record(position, seed=args.seed + index))
        else:
            train_rows.append(recovery_record(position, lexicon))
        preferences.append(preference_record(position, lexicon, seed=args.seed + index))
    train_rows.sort(key=lambda item: stable_key(args.seed + 2, str(item["id"])))

    validation_rows: list[dict[str, Any]] = []
    for index, position in enumerate(loss_positions):
        validation_rows.append(full_move_record(position))
        if index % 2 == 0:
            validation_rows.append(ranking_record(position, seed=args.seed + 10_000 + index))
        else:
            validation_rows.append(recovery_record(position, lexicon))
    validation_rows.sort(key=lambda item: stable_key(args.seed + 3, str(item["id"])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        "train.jsonl": write_jsonl(args.output_dir / "train.jsonl", train_rows),
        "validation.jsonl": write_jsonl(args.output_dir / "validation.jsonl", validation_rows),
        "preferences.jsonl": write_jsonl(args.output_dir / "preferences.jsonl", preferences),
        "train_positions.json": write_json(args.output_dir / "train_positions.json", selected),
        "validation_positions.json": write_json(args.output_dir / "validation_positions.json", loss_positions),
        "gate_positions.json": write_json(args.output_dir / "gate_positions.json", gate),
    }
    record_types = Counter(row["record_type"] for row in train_rows)
    manifest = {
        "version": "v4-pilot-1",
        "seed": args.seed,
        "source_dir": str(args.source_dir),
        "official_benchmark_policy": "hash exclusion only; never training or checkpoint selection",
        "split_unit": "complete self-play game",
        "training_board_encoding": "dense-grid",
        "evaluation_board_encodings": ["dense-grid", "sparse-benchmark-control"],
        "train": {
            "positions": len(selected),
            "records": len(train_rows),
            "preferences": len(preferences),
            "games": len({item["source_game_id"] for item in selected}),
            "openings": sum(int(item["band_ply"]) == 0 for item in selected),
            "non_openings": sum(int(item["band_ply"]) > 0 for item in selected),
            "record_types": dict(sorted(record_types.items())),
        },
        "validation": {
            "positions": len(loss_positions),
            "records": len(validation_rows),
            "games": len({item["source_game_id"] for item in loss_positions}),
        },
        "gate": {
            "positions": len(gate),
            "games": len({item["source_game_id"] for item in gate}),
            "openings": sum(int(item["band_ply"]) == 0 for item in gate),
            "by_ply": dict(sorted(Counter(str(item["band_ply"]) for item in gate).items())),
        },
        "cross_split_game_overlap": len(
            {item["source_game_id"] for item in selected}
            & {item["source_game_id"] for item in validation_source}
        ),
        "official_overlap": 0,
        "hashes": hashes,
    }
    manifest["manifest_sha256"] = write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
