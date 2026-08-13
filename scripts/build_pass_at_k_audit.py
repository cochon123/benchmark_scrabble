#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scrabble_bench.config import DATASET_PATH
from scrabble_bench.training import position_key
from scrabble_bench.v4_diagnostic import clean_placements, stable_key
from scrabble_bench.v41_training import free_record


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def game_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        if "selected_game_ids" in value:
            return {str(item) for item in value["selected_game_ids"]}
        if "gate_protocol" in value:
            return {str(item) for item in value["gate_protocol"].get("selected_games", [])}
        value = value.get("positions", [])
    return {
        str((item.get("position") or item).get("source_game_id"))
        for item in value
        if (item.get("position") or item).get("source_game_id")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/general_v3_merged/test_positions.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/general_v42_passk"))
    parser.add_argument("--boards", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8432)
    args = parser.parse_args()

    excluded = set()
    for path in (
        Path("data/general_v4_diagnostic/manifest.json"),
        Path("data/general_v42/manifest.json"),
        Path("artifacts/v3_test20_final/evaluations/test20/positions.json"),
    ):
        excluded |= game_ids(path)

    official = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    official_keys = {position_key(item) for item in official}
    source = json.loads(args.source.read_text(encoding="utf-8"))
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position in source:
        game = str(position["source_game_id"])
        if game not in excluded and int(position["band_ply"]) in {6, 10, 14}:
            by_game[game].append(position)

    selected_games = sorted(by_game, key=lambda item: stable_key(args.seed, item))[: args.boards]
    if len(selected_games) != args.boards:
        raise RuntimeError(f"Need {args.boards} untouched games, found {len(selected_games)}")

    positions = []
    for game in selected_games:
        candidates = sorted(by_game[game], key=lambda item: stable_key(args.seed + 1, item["id"]))
        positions.append(candidates[0])
    if any(position_key(item) in official_keys for item in positions):
        raise RuntimeError("Official benchmark overlap detected")

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "positions.json"
    data_path.write_text(json.dumps(records, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest = {
        "version": "v4.2-passk-audit-1",
        "seed": args.seed,
        "source": str(args.source),
        "boards": len(records),
        "whole_games": len(selected_games),
        "selected_game_ids": selected_games,
        "excluded_prior_gate_games": len(excluded),
        "official_overlap": 0,
        "train_overlap": 0,
        "openings": 0,
        "by_ply": dict(sorted(Counter(int(item["band_ply"]) for item in positions).items())),
        "positions_sha256": sha256(data_path),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
