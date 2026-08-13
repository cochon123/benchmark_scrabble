#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scrabble_bench.runner import prompt_for_position
from scrabble_bench.training import assistant_payload, position_key
from scrabble_bench.v4_diagnostic import stable_key
from scrabble_bench.v4_training import full_move_record


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--v42-rollouts", type=Path, required=True)
    parser.add_argument("--v41-rollouts", type=Path, required=True)
    parser.add_argument("--v42-rehearsal", type=Path, required=True)
    parser.add_argument("--v41-rehearsal", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9431)
    args = parser.parse_args()

    records = json.loads(args.positions.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in records}
    rollouts = []
    for label, path in (("v42", args.v42_rollouts), ("v41", args.v41_rollouts)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload["results"]:
            rollouts.append((label, result))

    train: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    for item in records:
        train.append(full_move_record(item["position"]))

    category_priority = {"collision": 0, "main_word": 1, "geometry": 2, "connection": 3}
    recovery_count = legal_count = optimal_count = 0
    for label, result in rollouts:
        source = by_id[result["id"]]
        position = source["position"]
        invalid = [sample for sample in result["samples"] if not sample["legal"] and sample["raw_response"]]
        invalid.sort(
            key=lambda sample: (
                category_priority.get(str(sample["error_category"]), 9),
                stable_key(args.seed, [label, result["id"], sample["raw_response"]]),
            )
        )
        selected = []
        categories = set()
        for sample in invalid:
            category = str(sample["error_category"])
            if category in categories:
                continue
            categories.add(category)
            selected.append(sample)
            if len(selected) == 2:
                break
        for index, sample in enumerate(selected):
            messages = prompt_for_position(
                position,
                {"raw_response": sample["raw_response"], "error": sample["error"] or "Rejected by verifier"},
                board_encoding="dense",
            )
            train.append(
                {
                    "id": f"{position['id']}--v43-{label}-recovery-{index}",
                    "source_id": position["id"],
                    "position_key": position_key(position),
                    "optimal_score": int(position["optimal_score"]),
                    "record_type": f"v43-{label}-onpolicy-recovery",
                    "error_category": sample["error_category"],
                    "messages": messages + [{"role": "assistant", "content": assistant_payload(position)}],
                }
            )
            preferences.append(
                {
                    "id": f"{position['id']}--v43-{label}-preference-{index}",
                    "source_id": position["id"],
                    "prompt": source["messages"],
                    "chosen": assistant_payload(position),
                    "rejected": sample["raw_response"],
                    "error_category": sample["error_category"],
                    "optimal_score": int(position["optimal_score"]),
                }
            )
            recovery_count += 1
        for sample in result["samples"]:
            if sample["legal"]:
                legal_count += 1
                optimal_count += bool(sample["optimal"])

    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    # Rehearsal protects the coordinate and plan skills that survived earlier gates.
    v42_rehearsal = read_jsonl(args.v42_rehearsal)
    v41_rehearsal = read_jsonl(args.v41_rehearsal)
    v42_rehearsal.sort(key=lambda item: stable_key(args.seed + 1, item["id"]))
    v41_rehearsal.sort(key=lambda item: stable_key(args.seed + 2, item["id"]))
    train.extend(v42_rehearsal[:200])
    train.extend(v41_rehearsal[:200])
    train.sort(key=lambda item: stable_key(args.seed + 3, item["id"]))

    validation = read_jsonl(args.validation)
    validation.sort(key=lambda item: stable_key(args.seed + 4, item["id"]))
    validation = validation[:100]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    preference_path = args.output_dir / "preferences.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    write_jsonl(preference_path, preferences)
    manifest = {
        "version": "v4.3-selftrain-1",
        "train_records": len(train),
        "validation_records": len(validation),
        "preference_pairs_reserved": len(preferences),
        "onpolicy_recovery_records": recovery_count,
        "legal_rollout_samples": legal_count,
        "optimal_rollout_samples": optimal_count,
        "record_types": dict(sorted(Counter(item["record_type"] for item in train).items())),
        "hashes": {
            "train": sha256(train_path),
            "validation": sha256(validation_path),
            "preferences": sha256(preference_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
