#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scrabble_bench.config import DATASET_PATH, resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.runner import prompt_for_position
from scrabble_bench.training import position_key
from scrabble_bench.v4_diagnostic import clean_placements, stable_key
from scrabble_bench.v41_training import move_plan


def intervention_messages(
    position: dict[str, Any], plan: dict[str, Any], task: str
) -> list[dict[str, str]]:
    messages = prompt_for_position(position, board_encoding="dense")
    user = json.loads(messages[-1]["content"])
    hints: dict[str, Any]
    if task == "word-only":
        hints = {"word": plan["word"]}
        instruction = "The oracle supplies the optimal main word. Locate and play its highest-scoring legal placement."
    elif task == "word-direction":
        hints = {"word": plan["word"], "direction": plan["direction"]}
        instruction = "The oracle supplies the optimal main word and direction. Locate and play its highest-scoring legal placement."
    elif task == "location":
        hints = {"start_row": plan["start_row"], "start_col": plan["start_col"]}
        instruction = "The oracle supplies the optimal main-word start square. Find the word and direction and play the highest-scoring legal move beginning there."
    elif task == "location-direction":
        hints = {
            "start_row": plan["start_row"],
            "start_col": plan["start_col"],
            "direction": plan["direction"],
        }
        instruction = "The oracle supplies the optimal main-word start and direction. Find and play the highest-scoring legal word for that slot."
    elif task == "complete-plan":
        hints = {
            "word": plan["word"],
            "start_row": plan["start_row"],
            "start_col": plan["start_col"],
            "direction": plan["direction"],
        }
        instruction = "The oracle supplies the complete optimal plan. Execute it as new-tile placements."
    else:
        raise ValueError(f"Unsupported intervention task: {task}")
    user["task_mode"] = f"V8_ORACLE_{task.upper().replace('-', '_')}"
    user["oracle_hint"] = hints
    user["instruction"] = instruction + " Reply with raw play_move JSON only."
    messages[-1]["content"] = json.dumps(user, separators=(",", ":"))
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boards", type=int, default=46)
    parser.add_argument("--seed", type=int, default=12801)
    args = parser.parse_args()

    lexicon_path = resolve_lexicon_path()
    lexicon = Lexicon.from_path(lexicon_path)
    positions = json.loads(args.positions.read_text(encoding="utf-8"))
    official = {position_key(row) for row in json.loads(DATASET_PATH.read_text())}
    by_game: dict[str, list[dict[str, Any]]] = {}
    for position in positions:
        if int(position["band_ply"]) <= 0:
            continue
        if position_key(position) in official:
            raise RuntimeError("Official benchmark overlap in oracle intervention source")
        by_game.setdefault(str(position["source_game_id"]), []).append(position)
    games = sorted(by_game, key=lambda game: stable_key(args.seed, game))[: args.boards]
    selected = [
        min(
            by_game[game],
            key=lambda row: (
                abs(int(row["band_ply"]) - (6, 10, 14)[index % 3]),
                stable_key(args.seed + 1, row["id"]),
            ),
        )
        for index, game in enumerate(games)
    ]
    tasks = ("word-only", "word-direction", "location", "location-direction", "complete-plan")
    records = []
    for position in selected:
        plan = move_plan(
            position,
            clean_placements(position["canonical_optimal_move"]["placements"]),
            lexicon,
        )
        for task in tasks:
            records.append(
                {
                    "id": f"{position['id']}--v8-{task}",
                    "source_id": position["id"],
                    "source_game_id": position["source_game_id"],
                    "band_ply": int(position["band_ply"]),
                    "task": task,
                    "position": position,
                    "oracle_plan": plan,
                    "messages": intervention_messages(position, plan, task),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")
    manifest = {
        "version": "v8-oracle-interventions-1",
        "boards": len(selected),
        "whole_games": len(games),
        "records": len(records),
        "tasks": list(tasks),
        "official_overlap": 0,
        "lexicon_path": str(lexicon_path),
        "lexicon_sha256": hashlib.sha256(lexicon_path.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
