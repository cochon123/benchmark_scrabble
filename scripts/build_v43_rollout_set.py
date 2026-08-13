#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scrabble_bench.config import DATASET_PATH
from scrabble_bench.training import position_key
from scrabble_bench.v4_diagnostic import clean_placements, stable_key
from scrabble_bench.v41_training import free_record


ROOT = Path("data/general_v43")
SEED = 9431


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = json.loads(Path("data/general_v3_merged/train_positions.json").read_text(encoding="utf-8"))
    prior = json.loads(Path("data/general_v4_pilot/train_positions.json").read_text(encoding="utf-8"))
    prior_keys = {position_key(item) for item in prior}
    official_keys = {
        position_key(item)
        for item in json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    }
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position in source:
        if int(position["band_ply"]) not in {6, 10, 14}:
            continue
        if position_key(position) in prior_keys or position_key(position) in official_keys:
            continue
        by_game[str(position["source_game_id"])].append(position)
    games = sorted(by_game, key=lambda game: stable_key(SEED, game))[:160]
    if len(games) != 160:
        raise RuntimeError(f"Need 160 eligible train games, found {len(games)}")
    positions = [
        sorted(by_game[game], key=lambda item: stable_key(SEED + 1, item["id"]))[0]
        for game in games
    ]
    records = []
    for position in positions:
        record = free_record(position)
        records.append(
            {
                "id": record["id"],
                "source_id": position["id"],
                "source_game_id": position["source_game_id"],
                "band_ply": int(position["band_ply"]),
                "messages": record["messages"][:-1],
                "position": position,
                "target_placements": clean_placements(position["canonical_optimal_move"]["placements"]),
            }
        )
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "rollout_positions.json"
    path.write_text(json.dumps(records, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest = {
        "version": "v4.3-rollout-1",
        "seed": SEED,
        "positions": len(records),
        "games": len(games),
        "split": "train-only",
        "prior_v4_position_overlap": 0,
        "official_overlap": 0,
        "by_ply": dict(sorted(Counter(item["band_ply"] for item in records).items())),
        "positions_sha256": sha256(path),
    }
    (ROOT / "rollout_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
