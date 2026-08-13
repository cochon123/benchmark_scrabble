#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.v4_diagnostic import stable_key
from scrabble_bench.v41_training import (
    anchor_direction_record,
    diagnostic_from_training_record,
    free_record,
    full_candidate_record,
    plan_copy_record,
    plan_ranking_record,
    word_direction_record,
)


ROOT = Path("data/general_v41")
SEED = 6419


def write(path: Path, value: Any, *, compact: bool = False) -> str:
    payload = (json.dumps(value, separators=(",", ":")) if compact else json.dumps(value, indent=2)) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def safe_build(builder: Callable[[], dict[str, Any]]) -> dict[str, Any] | None:
    try:
        return builder()
    except RuntimeError:
        return None


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    train = json.loads(Path("data/general_v4_pilot/train_positions.json").read_text())
    validation = json.loads(Path("data/general_v4_pilot/validation_positions.json").read_text())
    diagnostic = json.loads(Path("data/general_v4_diagnostic/diagnostic.json").read_text())
    diagnostic_positions = [item["position"] for item in diagnostic]
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    train.sort(key=lambda item: stable_key(SEED, item["id"]))

    stage1 = []
    modes = ("copy", "legality", "ranking", "anchor_direction")
    for index, position in enumerate(train):
        mode = modes[len(stage1) % len(modes)]
        row = safe_build(lambda p=position, t=mode, s=SEED + index: full_candidate_record(p, lexicon, task=t, seed=s))
        if row is not None:
            stage1.append(row)
        if len(stage1) == 1600:
            break

    stage2 = []
    for index, position in enumerate(train):
        builder = (
            (lambda p=position: plan_copy_record(p, lexicon))
            if index % 2 == 0
            else (lambda p=position, s=SEED + index: plan_ranking_record(p, lexicon, seed=s))
        )
        row = safe_build(builder)
        if row is not None:
            stage2.append(row)
        if len(stage2) == 2400:
            break

    stage3 = []
    builders = (
        lambda p, _: plan_copy_record(p, lexicon),
        lambda p, s: plan_ranking_record(p, lexicon, seed=s),
        lambda p, _: word_direction_record(p, lexicon),
        lambda p, _: anchor_direction_record(p, lexicon),
        lambda p, _: free_record(p),
    )
    for index, position in enumerate(train):
        row = safe_build(lambda p=position, s=SEED + index, b=builders[index % len(builders)]: b(p, s))
        if row is not None:
            stage3.append(row)
        if len(stage3) == 2400:
            break

    validation_rows = []
    for index, position in enumerate(validation[:200]):
        builder = builders[index % len(builders)]
        row = safe_build(lambda p=position, s=SEED + 20_000 + index, b=builder: b(p, s))
        if row is not None:
            validation_rows.append(row)

    candidate_gate = [item for item in diagnostic if item["task"] in {"copy", "ranking"}]
    plan_gate = []
    final_gate = []
    for index, position in enumerate(diagnostic_positions[:100]):
        builder = (
            (lambda p=position: plan_copy_record(p, lexicon))
            if index % 2 == 0
            else (lambda p=position, s=SEED + index: plan_ranking_record(p, lexicon, seed=s))
        )
        row = safe_build(builder)
        if row is not None:
            plan_gate.append(diagnostic_from_training_record(row, position))
    for index, position in enumerate(diagnostic_positions):
        builder = (
            (lambda p=position: word_direction_record(p, lexicon))
            if index % 4 == 0
            else (lambda p=position: anchor_direction_record(p, lexicon))
            if index % 4 == 1
            else (lambda p=position: free_record(p))
            if index % 4 == 2
            else (lambda p=position, s=SEED + index: plan_ranking_record(p, lexicon, seed=s))
        )
        row = safe_build(builder)
        if row is not None:
            final_gate.append(diagnostic_from_training_record(row, position))

    files = {
        "stage1.jsonl": write_jsonl(ROOT / "stage1.jsonl", stage1),
        "stage2.jsonl": write_jsonl(ROOT / "stage2.jsonl", stage2),
        "stage3.jsonl": write_jsonl(ROOT / "stage3.jsonl", stage3),
        "validation.jsonl": write_jsonl(ROOT / "validation.jsonl", validation_rows),
        "candidate_gate.json": write(ROOT / "candidate_gate.json", candidate_gate, compact=True),
        "plan_gate.json": write(ROOT / "plan_gate.json", plan_gate, compact=True),
        "final_gate.json": write(ROOT / "final_gate.json", final_gate, compact=True),
        "baseline_gate.json": write(ROOT / "baseline_gate.json", plan_gate + final_gate, compact=True),
    }
    manifest = {
        "version": "v4.1-curriculum-1",
        "seed": SEED,
        "init_adapter": "V4 step-600 adapter",
        "official_benchmark_policy": "excluded and untouched",
        "stages": {
            "stage1": {"records": len(stage1), "types": dict(Counter(row["record_type"] for row in stage1)), "steps": 200},
            "stage2": {"records": len(stage2), "types": dict(Counter(row["record_type"] for row in stage2)), "steps": 300},
            "stage3": {"records": len(stage3), "types": dict(Counter(row["record_type"] for row in stage3)), "steps": 300},
        },
        "validation_records": len(validation_rows),
        "gates": {"candidate": len(candidate_gate), "plan": len(plan_gate), "final": len(final_gate), "baseline": len(plan_gate + final_gate)},
        "hashes": files,
    }
    manifest["manifest_sha256"] = write(ROOT / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
