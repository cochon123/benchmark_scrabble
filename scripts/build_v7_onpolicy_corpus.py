#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.v4_diagnostic import stable_key
from scrabble_bench.v7_training import (
    col_record,
    correction_record,
    direction_record,
    location_record,
    plan_record,
    row_record,
    word_record,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dataset", type=Path, required=True)
    parser.add_argument("--rollout-evaluation", type=Path, required=True)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11703)
    args = parser.parse_args()
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    rollout_rows = json.loads(args.rollout_dataset.read_text(encoding="utf-8"))
    positions = {row["id"]: row["position"] for row in rollout_rows}
    evaluation = json.loads(args.rollout_evaluation.read_text(encoding="utf-8"))
    stage_a = read_jsonl(args.stage_a)

    onpolicy: list[dict[str, Any]] = []
    first_errors: Counter[str] = Counter()
    optimal_rollouts = 0
    legal_rollouts = 0
    for result in evaluation["results"]:
        position = positions[result["id"]]
        legal_rollouts += bool(result["legal"])
        optimal_rollouts += bool(result["optimal"])
        if result["optimal"]:
            positive = plan_record(position, lexicon)
            positive["id"] += "--onpolicy-positive"
            positive["record_type"] = "v7-onpolicy-positive"
            onpolicy.append(positive)
            continue
        first_error = str(result["first_error"] or "unknown")
        first_errors[first_error] += 1
        correction = correction_record(
            position,
            lexicon,
            previous_attempt=str(result["raw_response"]),
            first_error=first_error,
        )
        onpolicy.append(correction)
        direct = plan_record(position, lexicon)
        direct["id"] += "--onpolicy-direct"
        direct["record_type"] = "v7-onpolicy-direct"
        onpolicy.append(direct)
        builder = {
            "row": row_record,
            "column": col_record,
            "direction": direction_record,
            "word": word_record,
            "suboptimal_location": location_record,
        }.get(first_error)
        if builder is not None:
            component = builder(position, lexicon)
            component["id"] += "--onpolicy-focus"
            component["record_type"] += "-onpolicy-focus"
            onpolicy.append(component)

    factorized_replay = [row for row in stage_a if str(row["record_type"]).startswith("v7-")]
    legacy_replay = [row for row in stage_a if not str(row["record_type"]).startswith("v7-")]
    factorized_replay.sort(key=lambda row: stable_key(args.seed, row["id"]))
    legacy_replay.sort(key=lambda row: stable_key(args.seed + 1, row["id"]))
    train = onpolicy + factorized_replay[:600] + legacy_replay[:600]
    train.sort(key=lambda row: stable_key(args.seed + 2, row["id"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "stage_b.jsonl"
    train_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in train),
        encoding="utf-8",
    )
    validation_path = args.output_dir / "validation.jsonl"
    validation_path.write_bytes(args.validation.read_bytes())
    manifest = {
        "version": "v7-onpolicy-corrections-1",
        "rollout_positions": len(evaluation["results"]),
        "rollout_legal": legal_rollouts,
        "rollout_optimal": optimal_rollouts,
        "onpolicy_records": len(onpolicy),
        "factorized_replay_records": min(600, len(factorized_replay)),
        "legacy_replay_records": min(600, len(legacy_replay)),
        "train_records": len(train),
        "record_types": dict(sorted(Counter(row["record_type"] for row in train).items())),
        "first_errors": dict(sorted(first_errors.items())),
        "hashes": {
            "stage_b": sha256(train_path),
            "validation": sha256(validation_path),
            "rollout_evaluation": sha256(args.rollout_evaluation),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
