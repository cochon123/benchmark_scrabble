#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.v4_diagnostic import clean_placements, stable_key
from scrabble_bench.v41_training import free_record, plan_ranking_record
from scrabble_bench.v42_training import (
    anchor_mask_record,
    cross_check_record,
    start_candidates,
    start_choice_record,
    start_location_record,
    start_to_move_record,
    word_direction_move_record,
)


ROOT = Path("data/general_v42")
SEED = 7421


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    return {"records": len(rows), "sha256": sha256(path), "bytes": path.stat().st_size}


def write_json(path: Path, payload: Any) -> dict[str, Any]:
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return {"records": len(payload), "sha256": sha256(path), "bytes": path.stat().st_size}


def safe_build(builder: Callable[[], dict[str, Any]]) -> dict[str, Any] | None:
    try:
        return builder()
    except (RuntimeError, ValueError):
        return None


def game_ids_from(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("positions", payload) if isinstance(payload, dict) else payload
    ids = set()
    for row in rows:
        position = row.get("position", row)
        game = position.get("source_game_id") or row.get("source_game_id")
        if game:
            ids.add(str(game))
    return ids


def diagnostic(record: dict[str, Any], position: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    task = str(record["record_type"]).removeprefix("v42-").removeprefix("v41-")
    row = {
        "id": record["id"],
        "source_id": position["id"],
        "source_game_id": position["source_game_id"],
        "band_ply": int(position["band_ply"]),
        "task": task,
        "position": position,
        "messages": record["messages"][:-1],
        "target_content": record["messages"][-1]["content"],
        "target_placements": clean_placements(position["canonical_optimal_move"]["placements"]),
    }
    if task in {"start-choice", "start-location"}:
        _, targets = start_candidates(position, lexicon, seed=SEED + int(position["band_ply"]))
        row["target_starts"] = [list(item) for item in sorted(targets)]
        if task == "start-choice":
            prompt = json.loads(record["messages"][-2]["content"])
            row["candidate_starts"] = prompt["candidate_starts"]
    return row


def select_fresh_gate_positions(test_positions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded = game_ids_from(Path("data/general_v4_diagnostic/diagnostic.json"))
    excluded |= game_ids_from(Path("artifacts/v3_test20_final/evaluations/test20/positions.json"))
    by_game: dict[str, list[dict[str, Any]]] = {}
    for position in test_positions:
        game = str(position["source_game_id"])
        if game in excluded or int(position["band_ply"]) == 0:
            continue
        by_game.setdefault(game, []).append(position)
    games = sorted(by_game, key=lambda game: stable_key(SEED, game))[:30]
    if len(games) != 30:
        raise RuntimeError(f"Need 30 fresh gate games, found {len(games)}")
    selected = []
    for game in games:
        rows = sorted(by_game[game], key=lambda row: (int(row["band_ply"]), stable_key(SEED, row["id"])))
        if len(rows) < 10:
            raise RuntimeError(f"Gate game {game} has only {len(rows)} non-opening positions")
        selected.extend(rows[:10])
    return selected, {"excluded_games": len(excluded), "selected_games": games}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    train = json.loads(Path("data/general_v4_pilot/train_positions.json").read_text(encoding="utf-8"))
    test = json.loads(Path("data/general_v3_merged/test_positions.json").read_text(encoding="utf-8"))
    train.sort(key=lambda row: stable_key(SEED, row["id"]))

    stage1 = []
    for index, position in enumerate(train[:1500]):
        row = safe_build(lambda p=position, s=SEED + index: start_choice_record(p, lexicon, seed=s))
        if row:
            stage1.append(row)
    for position in train[:1500]:
        row = safe_build(lambda p=position: start_location_record(p, lexicon))
        if row:
            stage1.append(row)
    for index, position in enumerate(train[:700]):
        row = safe_build(lambda p=position, s=SEED + 10_000 + index: anchor_mask_record(p, seed=s))
        if row:
            stage1.append(row)
        if sum(item["record_type"] == "v42-anchor-mask" for item in stage1) == 500:
            break
    for index, position in enumerate(train):
        row = safe_build(lambda p=position, s=SEED + 20_000 + index: cross_check_record(p, lexicon, seed=s))
        if row:
            stage1.append(row)
        if sum(item["record_type"] == "v42-cross-check" for item in stage1) == 500:
            break

    stage2 = []
    for position in train[:1500]:
        row = safe_build(lambda p=position: start_to_move_record(p, lexicon))
        if row:
            stage2.append(row)
    for position in train[:1500]:
        row = safe_build(lambda p=position: word_direction_move_record(p, lexicon))
        if row:
            stage2.append(row)
    for position in train[1500:2500]:
        row = safe_build(lambda p=position: start_location_record(p, lexicon))
        if row:
            stage2.append(row)

    stage3 = []
    for index, position in enumerate(train):
        builders = (
            lambda p, _: free_record(p),
            lambda p, _: word_direction_move_record(p, lexicon),
            lambda p, s: plan_ranking_record(p, lexicon, seed=s),
            lambda p, _: start_location_record(p, lexicon),
        )
        row = safe_build(lambda p=position, s=SEED + 30_000 + index, b=builders[index % 4]: b(p, s))
        if row:
            stage3.append(row)
        if len(stage3) == 4000:
            break

    gate_positions, gate_manifest = select_fresh_gate_positions(test)
    locate_gate = []
    for index, position in enumerate(gate_positions[:100]):
        for builder in (
            lambda p=position, s=SEED + 40_000 + index: start_choice_record(p, lexicon, seed=s),
            lambda p=position: start_location_record(p, lexicon),
        ):
            row = safe_build(builder)
            if row:
                locate_gate.append(diagnostic(row, position, lexicon))
    word_gate = []
    for position in gate_positions[100:200]:
        row = safe_build(lambda p=position: word_direction_move_record(p, lexicon))
        if row:
            word_gate.append(diagnostic(row, position, lexicon))
    free_gate = []
    for position in gate_positions[200:300]:
        row = safe_build(lambda p=position: free_record(p))
        if row:
            free_gate.append(diagnostic(row, position, lexicon))

    files = {
        "stage1.jsonl": write_jsonl(ROOT / "stage1.jsonl", stage1),
        "stage2.jsonl": write_jsonl(ROOT / "stage2.jsonl", stage2),
        "stage3.jsonl": write_jsonl(ROOT / "stage3.jsonl", stage3),
        "validation.jsonl": write_jsonl(ROOT / "validation.jsonl", stage1[-100:]),
        "localization_gate.json": write_json(ROOT / "localization_gate.json", locate_gate),
        "word_move_gate.json": write_json(ROOT / "word_move_gate.json", word_gate),
        "free_gate.json": write_json(ROOT / "free_gate.json", free_gate),
        "baseline_gate.json": write_json(ROOT / "baseline_gate.json", locate_gate + word_gate + free_gate),
    }
    manifest = {
        "version": "v4.2-data-1",
        "seed": SEED,
        "train_source": "audited V4 pilot train positions",
        "gate_source": "fresh whole games from frozen V3 generated test",
        "gate_protocol": gate_manifest,
        "record_types": {
            filename: dict(sorted({
                row.get("record_type", row.get("task")): sum(
                    item.get("record_type", item.get("task")) == row.get("record_type", row.get("task"))
                    for item in rows
                )
                for row in rows
            }.items()))
            for filename, rows in {
                "stage1.jsonl": stage1,
                "stage2.jsonl": stage2,
                "stage3.jsonl": stage3,
                "validation.jsonl": stage1[-100:],
                "localization_gate.json": locate_gate,
                "word_move_gate.json": word_gate,
                "free_gate.json": free_gate,
            }.items()
        },
        "files": files,
    }
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = sha256(manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
