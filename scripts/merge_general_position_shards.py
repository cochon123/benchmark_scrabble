#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scrabble_bench.config import DATASET_PATH
from scrabble_bench.training import position_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/general_v2_merged"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    benchmark_keys = {position_key(position) for position in benchmark}
    split_positions: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    game_split: dict[str, str] = {}
    seen_keys = set(benchmark_keys)
    duplicates = 0

    for shard in args.shards:
        for split in split_positions:
            path = shard / f"{split}_positions.json"
            positions = json.loads(path.read_text(encoding="utf-8"))
            for position in positions:
                game_id = str(position["source_game_id"])
                prior_split = game_split.setdefault(game_id, split)
                if prior_split != split:
                    raise RuntimeError(
                        f"Game {game_id} appears in both {prior_split} and {split}"
                    )
                key = position_key(position)
                if key in seen_keys:
                    duplicates += 1
                    continue
                seen_keys.add(key)
                split_positions[split].append(position)

    manifest: dict[str, Any] = {
        "shards": [str(path) for path in args.shards],
        "benchmark_positions_excluded": len(benchmark_keys),
        "duplicate_positions_dropped": duplicates,
        "split_unit": "complete self-play game",
        "splits": {},
    }
    for split, positions in split_positions.items():
        positions.sort(key=lambda item: (int(item["source_seed"]), int(item["band_ply"])))
        payload = json.dumps(positions, indent=2)
        (args.output_dir / f"{split}_positions.json").write_text(
            payload,
            encoding="utf-8",
        )
        manifest["splits"][split] = {
            "games": len({item["source_game_id"] for item in positions}),
            "positions": len(positions),
            "positions_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "positions_by_ply": dict(
                sorted(Counter(int(item["band_ply"]) for item in positions).items())
            ),
        }

    validation_positions = split_positions["validation"]
    selection = [
        item
        for ply in sorted({int(row["band_ply"]) for row in validation_positions})
        for item in [
            item
            for item in validation_positions
            if int(item["band_ply"]) == ply
        ][:5]
    ]
    selection_payload = json.dumps(selection, indent=2)
    (args.output_dir / "selection_positions.json").write_text(
        selection_payload,
        encoding="utf-8",
    )
    manifest["selection"] = {
        "positions": len(selection),
        "positions_sha256": hashlib.sha256(selection_payload.encode()).hexdigest(),
        "per_ply": 5,
        "source_split": "validation",
    }

    train_positions = split_positions["train"]
    mining_selection = [
        item
        for ply in sorted({int(row["band_ply"]) for row in train_positions})
        for item in sorted(
            [row for row in train_positions if int(row["band_ply"]) == ply],
            key=lambda row: hashlib.sha256(str(row["id"]).encode()).hexdigest(),
        )[:10]
    ]
    mining_payload = json.dumps(mining_selection, indent=2)
    (args.output_dir / "train_mining_positions.json").write_text(
        mining_payload,
        encoding="utf-8",
    )
    manifest["train_mining_selection"] = {
        "positions": len(mining_selection),
        "positions_sha256": hashlib.sha256(mining_payload.encode()).hexdigest(),
        "per_ply": 10,
        "source_split": "train",
        "usage": "on-policy hard-negative collection only",
    }

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
