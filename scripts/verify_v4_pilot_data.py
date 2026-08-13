#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scrabble_bench.config import DATASET_PATH, resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.runner import parse_tool_payload
from scrabble_bench.training import position_key
from scrabble_bench.v4_training import verifier_reward


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently verify the v4 pilot corpus and reward.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/general_v4_pilot"))
    args = parser.parse_args()
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["hashes"].items():
        actual = sha256(args.data_dir / filename)
        if actual != expected:
            raise RuntimeError(f"Hash mismatch for {filename}: {actual} != {expected}")

    train = read_jsonl(args.data_dir / "train.jsonl")
    validation = read_jsonl(args.data_dir / "validation.jsonl")
    preferences = read_jsonl(args.data_dir / "preferences.jsonl")
    train_positions = json.loads((args.data_dir / "train_positions.json").read_text(encoding="utf-8"))
    validation_positions = json.loads((args.data_dir / "validation_positions.json").read_text(encoding="utf-8"))
    gate = json.loads((args.data_dir / "gate_positions.json").read_text(encoding="utf-8"))
    positions = {item["id"]: item for item in train_positions + validation_positions + gate}
    lexicon = Lexicon.from_path(resolve_lexicon_path())

    invalid_labels: list[str] = []
    for row in train + validation:
        completion = row["messages"][-1]["content"]
        position = positions[row["source_id"]]
        reward = verifier_reward(position, completion, lexicon)
        if not reward["legal"] or not reward["optimal"]:
            invalid_labels.append(str(row["id"]))
        parse_tool_payload(completion)
    if invalid_labels:
        raise RuntimeError(f"Invalid SFT labels: {invalid_labels[:10]}")

    preference_failures: list[str] = []
    preference_categories: Counter[str] = Counter()
    for row in preferences:
        position = positions[row["source_id"]]
        chosen = verifier_reward(position, row["chosen"], lexicon)
        rejected = verifier_reward(position, row["rejected"], lexicon)
        preference_categories[str(row["error_category"])] += 1
        if not chosen["optimal"] or chosen["reward"] <= rejected["reward"]:
            preference_failures.append(str(row["id"]))
    if preference_failures:
        raise RuntimeError(f"Invalid preference ordering: {preference_failures[:10]}")

    train_games = {item["source_game_id"] for item in train_positions}
    validation_games = {item["source_game_id"] for item in validation_positions + gate}
    loss_games = {item["source_game_id"] for item in validation_positions}
    gate_games = {item["source_game_id"] for item in gate}
    official_keys = {
        position_key(item)
        for item in json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    }
    all_keys = {position_key(item) for item in train_positions + validation_positions + gate}
    checks = {
        "passed": True,
        "train_records": len(train),
        "validation_records": len(validation),
        "preferences": len(preferences),
        "gate_positions": len(gate),
        "invalid_sft_labels": 0,
        "invalid_preferences": 0,
        "preference_categories": dict(sorted(preference_categories.items())),
        "train_validation_game_overlap": len(train_games & validation_games),
        "loss_gate_game_overlap": len(loss_games & gate_games),
        "official_position_overlap": len(official_keys & all_keys),
        "gate_openings": sum(int(item["band_ply"]) == 0 for item in gate),
    }
    if any(
        checks[key]
        for key in (
            "train_validation_game_overlap",
            "loss_gate_game_overlap",
            "official_position_overlap",
            "gate_openings",
        )
    ):
        checks["passed"] = False
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
