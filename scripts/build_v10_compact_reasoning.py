#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from scrabble_bench.config import DATASET_PATH, resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.training import position_key
from scrabble_bench.v10_training import compact_reasoning_record, teacher_reasoning_record


ROOT = Path(__file__).resolve().parents[1]
SEED = 10101


def choose_balanced(
    positions: list[dict[str, Any]], limit: int, seed: int, *, distinct_games: bool = False
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pools: dict[int, list[dict[str, Any]]] = {}
    for row in positions:
        if not row.get("board"):
            continue
        band = min((4, 7, 10, 14), key=lambda target: abs(int(row["band_ply"]) - target))
        pools.setdefault(band, []).append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    selected: list[dict[str, Any]] = []
    seen_games: set[str] = set()
    while len(selected) < limit:
        progressed = False
        for band in (4, 7, 10, 14):
            while pools.get(band):
                row = pools[band].pop()
                game = str(row["source_game_id"])
                if distinct_games and game in seen_games:
                    continue
                selected.append(row)
                seen_games.add(game)
                progressed = True
                break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    if len(selected) != limit:
        raise RuntimeError(f"Requested {limit} positions, selected {len(selected)}")
    return selected


def load_last_teacher_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    by_id: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_id[str(row["id"])] = row
    return list(by_id.values())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/v10_compact_reasoning")
    parser.add_argument("--train-limit", type=int, default=800)
    parser.add_argument("--validation-limit", type=int, default=80)
    parser.add_argument("--gate-limit", type=int, default=20)
    parser.add_argument("--teacher-replicas", type=int, default=6)
    parser.add_argument(
        "--teacher-traces",
        type=Path,
        default=ROOT / "artifacts/v10_compact_reasoning/teacher_traces.jsonl",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_dir = ROOT / "data/slot_pilot_v1"
    train_all = json.loads((source_dir / "train_positions.json").read_text(encoding="utf-8"))
    validation_all = json.loads(
        (source_dir / "validation_positions.json").read_text(encoding="utf-8")
    )
    test_all = json.loads((source_dir / "test_positions.json").read_text(encoding="utf-8"))
    train = choose_balanced(train_all, args.train_limit, SEED)
    validation = choose_balanced(validation_all, args.validation_limit, SEED + 1)
    gate = choose_balanced(test_all, args.gate_limit, SEED + 2, distinct_games=True)

    official_keys = {
        position_key(row) for row in json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    }
    train_games = {str(row["source_game_id"]) for row in train}
    validation_games = {str(row["source_game_id"]) for row in validation}
    gate_games = {str(row["source_game_id"]) for row in gate}
    if train_games & validation_games or train_games & gate_games or validation_games & gate_games:
        raise RuntimeError("Whole-game split overlap")
    if any(position_key(row) in official_keys for row in train + validation + gate):
        raise RuntimeError("Official benchmark overlap")

    lexicon = Lexicon.from_path(resolve_lexicon_path())
    train_rows = [
        compact_reasoning_record(row, lexicon, transform=transform)
        for row in train
        for transform in ("identity", "transpose")
    ]
    validation_rows = [
        compact_reasoning_record(row, lexicon, transform="identity") for row in validation
    ]

    positions_by_id = {str(row["id"]): row for row in train_all}
    accepted_teachers = []
    for teacher in load_last_teacher_rows(args.teacher_traces):
        if teacher.get("error") is not None or float(teacher.get("score_ratio", 0)) < 0.5:
            continue
        position = positions_by_id.get(str(teacher["id"]))
        if position is None:
            continue
        for replica in range(args.teacher_replicas):
            accepted_teachers.append(
                teacher_reasoning_record(position, str(teacher["response"]), replica=replica)
            )
    train_rows.extend(accepted_teachers)
    random.Random(SEED).shuffle(train_rows)

    hashes = {
        "train": write_jsonl(args.output_dir / "train.jsonl", train_rows),
        "validation": write_jsonl(args.output_dir / "validation.jsonl", validation_rows),
    }
    gate_payload = json.dumps(gate, separators=(",", ":"))
    (args.output_dir / "gate_positions.json").write_text(gate_payload, encoding="utf-8")
    hashes["gate"] = hashlib.sha256(gate_payload.encode()).hexdigest()
    manifest = {
        "version": "v10-thinking-distillation-pilot-1",
        "seed": SEED,
        "base_model": "Qwen/Qwen3-4B-Thinking-2507",
        "train_positions": len(train),
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "gate_positions": len(gate),
        "teacher_records": len(accepted_teachers),
        "teacher_unique_positions": len(accepted_teachers) // args.teacher_replicas,
        "train_types": dict(sorted(Counter(row["record_type"] for row in train_rows).items())),
        "train_validation_game_overlap": 0,
        "train_gate_game_overlap": 0,
        "official_positions_used": 0,
        "deployment_max_new_tokens": 512,
        "hashes": hashes,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
