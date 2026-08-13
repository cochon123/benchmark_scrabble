#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scrabble_bench.training import position_key


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6-train", type=Path, required=True)
    parser.add_argument("--v8-train", type=Path, required=True)
    parser.add_argument("--v8-validation", type=Path, required=True)
    parser.add_argument("--v8-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = load(args.v6_train) + load(args.v8_train)
    validation = load(args.v8_validation)
    test = load(args.v8_test)
    train_keys = {position_key(row) for row in train}
    validation_keys = {position_key(row) for row in validation}
    test_keys = {position_key(row) for row in test}
    if train_keys & validation_keys or train_keys & test_keys or validation_keys & test_keys:
        raise SystemExit("Position-key overlap detected between slot pilot splits")
    if len({row["source_game_id"] for row in train} & {row["source_game_id"] for row in test}):
        raise SystemExit("Game overlap detected between slot pilot train and test")
    if len({row["source_game_id"] for row in validation} & {row["source_game_id"] for row in test}):
        raise SystemExit("Game overlap detected between slot pilot validation and test")
    args.output.mkdir(parents=True, exist_ok=True)
    split_paths = {"train": args.output / "train_positions.json", "validation": args.output / "validation_positions.json", "test": args.output / "test_positions.json"}
    split_rows = {"train": train, "validation": validation, "test": test}
    for name, path in split_paths.items():
        path.write_text(json.dumps(split_rows[name], separators=(",", ":")) + "\n", encoding="utf-8")
    manifest = {
        "protocol": "Slot-first 450-class localization pilot",
        "training_reuse": "general_v6 train plus general_v8 fresh train; no slot model was trained on either",
        "validation_source": str(args.v8_validation), "test_source": str(args.v8_test),
        "split_policy": "whole-game-separated; identity-only; no official benchmark positions",
        "splits": {
            name: {
                "positions": len(rows), "games": len({row["source_game_id"] for row in rows}),
                "sha256": digest(path),
            }
            for name, rows, path in ((name, split_rows[name], split_paths[name]) for name in split_paths)
        },
        "source_hashes": {str(path): digest(path) for path in (args.v6_train, args.v8_train, args.v8_validation, args.v8_test)},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
