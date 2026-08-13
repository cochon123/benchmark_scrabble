#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "general_v5_8b"
SEED = 8501

TARGETS = {
    "v42-start-choice": 400,
    "v42-start-location": 400,
    "v42-anchor-mask": 200,
    "v42-cross-check": 200,
    "v41-plan-copy": 500,
    "v41-plan-ranking": 500,
    "v41-copy": 100,
    "v41-legality": 100,
    "v41-ranking": 100,
    "v41-anchor_direction": 100,
    "v41-word-direction": 120,
    "v41-anchor-direction": 120,
    "v41-free": 160,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def canonical_hash(rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def game_id(row: dict[str, Any]) -> str:
    if row.get("source_game_id"):
        return str(row["source_game_id"])
    source_id = str(row.get("source_id", row["id"].split("--", 1)[0]))
    return source_id.rsplit("-", 1)[0]


def main() -> None:
    rng = random.Random(SEED)
    sources = [
        ROOT / "data/general_v42/stage1.jsonl",
        ROOT / "data/general_v41/stage1.jsonl",
        ROOT / "data/general_v41/stage2.jsonl",
        ROOT / "data/general_v41/stage3.jsonl",
    ]
    validation_sources = [
        ROOT / "data/general_v42/validation.jsonl",
        ROOT / "data/general_v41/validation.jsonl",
    ]
    validation_all = [row for path in validation_sources for row in read_jsonl(path)]
    validation_ids = {row["id"] for row in validation_all}
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_source_ids: set[str] = set()
    for path in sources:
        for row in read_jsonl(path):
            if row["id"] not in validation_ids and row["id"] not in seen_source_ids:
                pools[str(row["record_type"])].append(row)
                seen_source_ids.add(row["id"])

    train: list[dict[str, Any]] = []
    for record_type, count in TARGETS.items():
        candidates = pools[record_type]
        if len(candidates) < count:
            raise RuntimeError(f"Need {count} {record_type} rows, found {len(candidates)}")
        train.extend(rng.sample(candidates, count))
    rng.shuffle(train)

    # Keep a compact, type-balanced loss-only validation set and never train on its IDs.
    grouped_validation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validation_all:
        grouped_validation[str(row["record_type"])].append(row)
    validation: list[dict[str, Any]] = []
    for record_type in sorted(grouped_validation):
        validation.extend(grouped_validation[record_type][:20])

    train_ids = [row["id"] for row in train]
    if len(train_ids) != len(set(train_ids)):
        raise RuntimeError("Duplicate train IDs")
    if set(train_ids) & {row["id"] for row in validation}:
        raise RuntimeError("Train/validation ID overlap")

    gate_ids: set[str] = set()
    for path in (
        ROOT / "data/general_v42/localization_gate.json",
        ROOT / "data/general_v41/plan_gate.json",
        ROOT / "data/general_v42_passk/positions.json",
    ):
        gate_ids.update(row["id"] for row in json.loads(path.read_text(encoding="utf-8")))
    if set(train_ids) & gate_ids:
        raise RuntimeError("Train/frozen-gate ID overlap")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT / "train.jsonl"
    validation_path = OUTPUT / "validation.jsonl"
    train_path.write_text("".join(json.dumps(row) + "\n" for row in train), encoding="utf-8")
    validation_path.write_text("".join(json.dumps(row) + "\n" for row in validation), encoding="utf-8")
    manifest = {
        "version": "v5-8b-capacity-pilot-1",
        "seed": SEED,
        "model": "Qwen/Qwen3-8B",
        "train_records": len(train),
        "validation_records": len(validation),
        "train_types": dict(sorted(Counter(row["record_type"] for row in train).items())),
        "validation_types": dict(sorted(Counter(row["record_type"] for row in validation).items())),
        "distinct_train_games": len({game_id(row) for row in train}),
        "train_validation_overlap": 0,
        "train_frozen_gate_overlap": 0,
        "official_positions_used": 0,
        "hashes": {"train": canonical_hash(train), "validation": canonical_hash(validation)},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
