#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from scrabble_bench.config import DATASET_PATH, resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.training import position_key, transform_position
from scrabble_bench.v4_diagnostic import stable_key
from scrabble_bench.v7_training import (
    col_record,
    diagnostic_record,
    factorized_plan,
    location_record,
    plan_record,
    plan_to_move_record,
    row_record,
    word_record,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/general_v7"
SEED = 10703


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {"records": len(rows), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
    return {"records": len(rows), "bytes": path.stat().st_size, "sha256": sha256(path)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def balanced_select(
    positions: list[dict[str, Any]],
    lexicon: Lexicon,
    *,
    count: int,
    seed: int,
    component: str,
) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for position in positions:
        try:
            plan = factorized_plan(position, lexicon)
        except (RuntimeError, ValueError):
            continue
        if component == "row":
            key: Any = plan["row"]
        elif component == "col":
            key = plan["col"]
        elif component == "location":
            key = (plan["row"], plan["col"])
        else:
            key = int(position["band_ply"])
        groups[key].append(position)
    for key, rows in groups.items():
        rows.sort(key=lambda row, k=key: stable_key(seed, [k, row["id"]]))
    keys = sorted(groups, key=lambda key: stable_key(seed + 1, key))
    chosen: list[dict[str, Any]] = []
    cursor = 0
    while len(chosen) < count and keys:
        key = keys[cursor % len(keys)]
        rows = groups[key]
        if rows:
            chosen.append(rows.pop())
        if not rows:
            keys.remove(key)
            cursor -= 1
        cursor += 1
    if len(chosen) != count:
        raise RuntimeError(f"Could select only {len(chosen)}/{count} positions for {component}")
    return chosen


def build_records(
    positions: list[dict[str, Any]],
    lexicon: Lexicon,
    builder: Callable[[dict[str, Any], Lexicon], dict[str, Any]],
    *,
    count: int,
    seed: int,
    component: str,
) -> list[dict[str, Any]]:
    selected = balanced_select(positions, lexicon, count=count, seed=seed, component=component)
    records = []
    for index, source in enumerate(selected):
        transform = "transpose" if index % 2 else "identity"
        position = transform_position(source, transform) if transform != "identity" else source
        record = builder(position, lexicon)
        record["source_id"] = source["id"]
        record["transform"] = transform
        records.append(record)
    return records


def choose_rollout_positions(
    positions: list[dict[str, Any]], games: set[str], lexicon: Lexicon
) -> list[dict[str, Any]]:
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position in positions:
        game = str(position["source_game_id"])
        if game in games and int(position["band_ply"]) > 0:
            by_game[game].append(position)
    selected = []
    for game in sorted(games):
        rows = sorted(
            by_game[game],
            key=lambda row: (abs(int(row["band_ply"]) - 9), stable_key(SEED, row["id"])),
        )[:4]
        for position in rows:
            selected.append(diagnostic_record(position, lexicon, task="free-plan"))
    if len(selected) != 160:
        raise RuntimeError(f"Expected 160 rollout positions, got {len(selected)}")
    return selected


def choose_gate(test: list[dict[str, Any]], lexicon: Lexicon) -> tuple[list[dict[str, Any]], list[str]]:
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position in test:
        if int(position["band_ply"]) > 0:
            by_game[str(position["source_game_id"])].append(position)
    games = sorted(by_game, key=lambda game: stable_key(SEED + 5000, game))[:30]
    rows = []
    tasks = ("row", "column", "location", "word", "free-plan")
    for index, game in enumerate(games):
        desired_ply = (4, 8, 12)[index % 3]
        position = min(
            by_game[game],
            key=lambda row: (abs(int(row["band_ply"]) - desired_ply), stable_key(SEED, row["id"])),
        )
        rows.extend(diagnostic_record(position, lexicon, task=task) for task in tasks)
    if len(rows) != 150:
        raise RuntimeError(f"Expected 150 gate records, got {len(rows)}")
    return rows, games


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    train = json.loads((ROOT / "data/general_v6_positions/train_positions.json").read_text())
    validation = json.loads((ROOT / "data/general_v6_positions/validation_positions.json").read_text())
    test = json.loads((ROOT / "data/general_v6_positions/test_positions.json").read_text())

    train_games = sorted({str(row["source_game_id"]) for row in train}, key=lambda game: stable_key(SEED, game))
    rollout_games = set(train_games[-40:])
    stage_a_positions = [row for row in train if str(row["source_game_id"]) not in rollout_games]

    stage_a: list[dict[str, Any]] = []
    specs = (
        (row_record, 650, "row"),
        (col_record, 650, "col"),
        (location_record, 650, "location"),
        (plan_record, 450, "general"),
        (word_record, 250, "general"),
        (plan_to_move_record, 250, "general"),
    )
    for index, (builder, count, component) in enumerate(specs):
        stage_a.extend(
            build_records(
                stage_a_positions,
                lexicon,
                builder,
                count=count,
                seed=SEED + 100 * index,
                component=component,
            )
        )

    v42_replay = [
        row for row in read_jsonl(ROOT / "data/general_v42/stage1.jsonl")
        if row["record_type"] in {"v42-start-choice", "v42-start-location"}
    ]
    v41_replay = [
        row for row in read_jsonl(ROOT / "data/general_v41/stage2.jsonl")
        if row["record_type"] in {"v41-plan-copy", "v41-plan-ranking", "v41-free"}
    ]
    v42_replay.sort(key=lambda row: stable_key(SEED + 2000, row["id"]))
    v41_replay.sort(key=lambda row: stable_key(SEED + 3000, row["id"]))
    stage_a.extend(v42_replay[:300])
    stage_a.extend(v41_replay[:300])
    stage_a.sort(key=lambda row: stable_key(SEED + 4000, row["id"]))

    validation_rows: list[dict[str, Any]] = []
    validation_specs = (
        (row_record, "row"),
        (col_record, "col"),
        (location_record, "location"),
        (word_record, "general"),
        (plan_record, "general"),
    )
    for index, (builder, component) in enumerate(validation_specs):
        validation_rows.extend(
            build_records(
                validation,
                lexicon,
                builder,
                count=20,
                seed=SEED + 6000 + index,
                component=component,
            )
        )
    validation_rows.sort(key=lambda row: stable_key(SEED + 7000, row["id"]))

    rollout = choose_rollout_positions(train, rollout_games, lexicon)
    gate, gate_games = choose_gate(test, lexicon)

    official_keys = {position_key(row) for row in json.loads(DATASET_PATH.read_text())}
    stage_a_source_games = {str(row["source_game_id"]) for row in stage_a_positions}
    validation_games = {str(row["source_game_id"]) for row in validation}
    test_games = {str(row["source_game_id"]) for row in test}
    if stage_a_source_games & rollout_games:
        raise RuntimeError("Stage-A/rollout game leakage")
    if (stage_a_source_games | rollout_games) & (validation_games | test_games):
        raise RuntimeError("V7 train/evaluation game leakage")
    if validation_games & test_games:
        raise RuntimeError("V7 validation/test game leakage")
    if any(position_key(row) in official_keys for row in train + validation + test):
        raise RuntimeError("Official benchmark position overlap")

    files = {
        "stage_a.jsonl": write_jsonl(OUTPUT / "stage_a.jsonl", stage_a),
        "validation.jsonl": write_jsonl(OUTPUT / "validation.jsonl", validation_rows),
        "rollout_gate.json": write_json(OUTPUT / "rollout_gate.json", rollout),
        "factorized_gate.json": write_json(OUTPUT / "factorized_gate.json", gate),
    }
    manifest = {
        "version": "v7-factorized-onpolicy-1",
        "seed": SEED,
        "representation": "ROW_A..O|COL_A..O|DIR_A_OR_D|WORD_TEXT",
        "stage_a_records": len(stage_a),
        "validation_records": len(validation_rows),
        "rollout_positions": len(rollout),
        "rollout_games": len(rollout_games),
        "factorized_gate_records": len(gate),
        "factorized_gate_games": len(gate_games),
        "factorized_gate_game_ids": gate_games,
        "record_types": dict(sorted(Counter(row["record_type"] for row in stage_a).items())),
        "coordinate_histograms": {
            "row": dict(sorted(Counter(factorized_plan(row, lexicon)["row"] for row in stage_a_positions).items())),
            "col": dict(sorted(Counter(factorized_plan(row, lexicon)["col"] for row in stage_a_positions).items())),
        },
        "contamination": {
            "official_positions_used": 0,
            "stage_a_rollout_game_overlap": 0,
            "train_validation_game_overlap": 0,
            "train_test_game_overlap": 0,
        },
        "files": files,
    }
    token_audit_path = OUTPUT / "token_audit.json"
    if token_audit_path.exists():
        token_audit = json.loads(token_audit_path.read_text(encoding="utf-8"))
        manifest["token_audit"] = {
            "max_length": token_audit["max_length"],
            "observed_max": token_audit["total"]["max"],
            "over_limit": token_audit["over_limit"],
            "passed": token_audit["passed"],
            "sha256": sha256(token_audit_path),
        }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
