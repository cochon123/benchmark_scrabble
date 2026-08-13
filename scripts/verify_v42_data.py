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
from scrabble_bench.solver import _cross_check_letters, grid_from_position, neighbors, validate_and_score_move
from scrabble_bench.training import position_key
from scrabble_bench.v42_training import all_start_outcomes


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/general_v42"))
    parser.add_argument("--train-positions", type=Path, default=Path("data/general_v4_pilot/train_positions.json"))
    parser.add_argument("--test-positions", type=Path, default=Path("data/general_v3_merged/test_positions.json"))
    args = parser.parse_args()
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        actual = sha256(args.data_dir / filename)
        if actual != expected["sha256"]:
            raise RuntimeError(f"Hash mismatch for {filename}: {actual} != {expected['sha256']}")
    train_positions = json.loads(args.train_positions.read_text(encoding="utf-8"))
    test_positions = json.loads(args.test_positions.read_text(encoding="utf-8"))
    positions = {str(item["id"]): item for item in train_positions + test_positions}
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    rows = []
    for filename in ("stage1.jsonl", "stage2.jsonl", "stage3.jsonl"):
        rows.extend(read_jsonl(args.data_dir / filename))

    failures = []
    types: Counter[str] = Counter()
    checked_outcomes: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        task = str(row["record_type"])
        types[task] += 1
        position = positions[str(row["source_id"])]
        target = row["messages"][-1]["content"]
        try:
            if task == "v42-start-choice":
                chosen = int(json.loads(target)["candidate"])
                prompt = json.loads(row["messages"][-2]["content"])
                selected = prompt["candidate_starts"][chosen]
                outcomes = checked_outcomes.setdefault(position["id"], all_start_outcomes(position, lexicon))
                match = next(item for item in outcomes if item["row"] == selected["row"] and item["col"] == selected["col"])
                if not match["legal"] or int(match["score"]) != int(position["optimal_score"]):
                    raise ValueError("Selected start is not verifier-optimal")
                if any(key in row["messages"][-2]["content"] for key in ('"legal"', '"score"', '"target"')):
                    raise ValueError("Prompt leaks a verifier label")
            elif task == "v42-start-location":
                selected = json.loads(target)
                outcomes = checked_outcomes.setdefault(position["id"], all_start_outcomes(position, lexicon))
                match = next(
                    item for item in outcomes
                    if item["row"] == int(selected["start_row"]) and item["col"] == int(selected["start_col"])
                )
                if not match["legal"] or int(match["score"]) != int(position["optimal_score"]):
                    raise ValueError("Location target is not verifier-optimal")
            elif task == "v42-anchor-mask":
                payload = json.loads(target)
                prompt = json.loads(row["messages"][-2]["content"])
                grid = grid_from_position(position["board"])
                expected = []
                for item in prompt["candidate_squares"]:
                    r, c = int(item["row"]), int(item["col"])
                    is_anchor = (r, c) == (7, 7) if not position["board"] else any(
                        grid[nr][nc] is not None for nr, nc in neighbors(r, c)
                    )
                    if is_anchor:
                        expected.append(int(item["candidate"]))
                if payload["anchor_candidates"] != expected:
                    raise ValueError("Anchor mask mismatch")
            elif task == "v42-cross-check":
                payload = json.loads(target)
                prompt = json.loads(row["messages"][-2]["content"])
                square = prompt["square"]
                allowed = _cross_check_letters(
                    grid_from_position(position["board"]),
                    str(prompt["main_direction"]),
                    int(square["row"]),
                    int(square["col"]),
                    lexicon.words,
                )
                if payload["allowed_letters"] != "".join(sorted(allowed)):
                    raise ValueError("Cross-check set mismatch")
            else:
                payload = parse_tool_payload(target)
                move = validate_and_score_move(
                    lexicon,
                    grid_from_position(position["board"]),
                    str(position["rack"]),
                    payload["arguments"]["placements"],
                )
                if int(move.score) != int(position["optimal_score"]):
                    raise ValueError("Move target is not optimal")
        except Exception as error:
            failures.append({"id": row["id"], "error": str(error)})
    if failures:
        raise RuntimeError(json.dumps(failures[:10], indent=2))

    gates = []
    for filename in ("localization_gate.json", "word_move_gate.json", "free_gate.json"):
        gates.extend(json.loads((args.data_dir / filename).read_text(encoding="utf-8")))
    train_games = {str(item["source_game_id"]) for item in train_positions}
    gate_games = {str(item["source_game_id"]) for item in gates}
    official_keys = {position_key(item) for item in json.loads(DATASET_PATH.read_text(encoding="utf-8"))}
    gate_keys = {position_key(item["position"]) for item in gates}
    checks = {
        "passed": True,
        "training_records": len(rows),
        "record_types": dict(sorted(types.items())),
        "invalid_labels": 0,
        "gate_records": len(gates),
        "gate_games": len(gate_games),
        "train_gate_game_overlap": len(train_games & gate_games),
        "official_gate_overlap": len(official_keys & gate_keys),
        "gate_openings": sum(int(item["position"]["band_ply"]) == 0 for item in gates),
        "recomputed_start_positions": len(checked_outcomes),
    }
    if checks["train_gate_game_overlap"] or checks["official_gate_overlap"] or checks["gate_openings"]:
        checks["passed"] = False
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
