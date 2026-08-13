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
from scrabble_bench.v4_diagnostic import diagnostic_record, stable_key


TASKS = ("copy", "legality", "ranking", "anchor_direction")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a held-out V4 ability-decomposition diagnostic.")
    parser.add_argument("--source", type=Path, default=Path("data/general_v3_merged/test_positions.json"))
    parser.add_argument("--exclude", type=Path, default=Path("artifacts/v3_test20_final/evaluations/test20/positions.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/general_v4_diagnostic"))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--positions-per-game", type=int, default=10)
    parser.add_argument("--seed", type=int, default=5519)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    excluded = json.loads(args.exclude.read_text(encoding="utf-8")) if args.exclude.exists() else []
    excluded_games = {str(item["source_game_id"]) for item in excluded}
    official_keys = {position_key(item) for item in json.loads(DATASET_PATH.read_text(encoding="utf-8"))}
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position in source:
        game = str(position["source_game_id"])
        if game not in excluded_games and int(position["band_ply"]) > 0:
            by_game[game].append(position)
    eligible_games = [game for game, rows in by_game.items() if len(rows) >= args.positions_per_game]
    eligible_games.sort(key=lambda game: stable_key(args.seed, game))
    selected_games = eligible_games[: args.games]
    if len(selected_games) != args.games:
        raise RuntimeError("Not enough untouched whole games for the diagnostic")

    selected: list[dict[str, Any]] = []
    for game in selected_games:
        rows = sorted(by_game[game], key=lambda item: stable_key(args.seed + 1, item["id"]))
        selected.extend(rows[: args.positions_per_game])
    selected.sort(key=lambda item: stable_key(args.seed + 2, item["id"]))
    if any(position_key(item) in official_keys for item in selected):
        raise RuntimeError("Official benchmark overlap detected")

    lexicon = Lexicon.from_path(resolve_lexicon_path())
    records = [
        diagnostic_record(position, lexicon, task=TASKS[index % len(TASKS)], seed=args.seed + index)
        for index, position in enumerate(selected)
    ]
    counts = Counter(record["task"] for record in records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "diagnostic.json"
    data_path.write_text(json.dumps(records, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest = {
        "version": "v4-diagnostic-1",
        "seed": args.seed,
        "source_split": str(args.source),
        "official_benchmark_policy": "excluded and untouched",
        "previous_test20_games_excluded": len(excluded_games),
        "games": len(selected_games),
        "positions": len(selected),
        "positions_per_game": args.positions_per_game,
        "tasks": dict(sorted(counts.items())),
        "openings": sum(int(item["band_ply"]) == 0 for item in selected),
        "official_overlap": 0,
        "selected_game_ids": selected_games,
        "position_ids_sha256": hashlib.sha256("\n".join(item["id"] for item in selected).encode()).hexdigest(),
        "diagnostic_sha256": sha256(data_path),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
