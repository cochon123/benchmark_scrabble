#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path


ROOT = Path("/content/v7-audit")
ARCHIVE = Path("/content/source.tar.gz")
JOB_MANIFEST = Path("/content/job_manifest.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads(JOB_MANIFEST.read_text(encoding="utf-8"))
    if sha256(ARCHIVE) != manifest["source_sha256"]:
        raise RuntimeError("Packaged source hash does not match the job manifest")
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(ROOT, filter="data")
    data = ROOT / "data/general_v7"
    data_manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in data_manifest["files"].items():
        path = data / name
        if sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Data hash mismatch: {name}")

    stage_a = [
        json.loads(line)
        for line in (data / "stage_a.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    validation = [
        json.loads(line)
        for line in (data / "validation.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    gate = json.loads((data / "factorized_gate.json").read_text(encoding="utf-8"))
    rollout = json.loads((data / "rollout_gate.json").read_text(encoding="utf-8"))
    grammar = re.compile(r"^ROW_[A-O]\|COL_[A-O]\|DIR_[AD]\|WORD_[A-Z]+$")
    free_targets = [
        row["target_content"] for row in gate if row["task"] == "free-plan"
    ]
    if not free_targets or not all(grammar.fullmatch(target) for target in free_targets):
        raise RuntimeError("Factorized free-plan targets violate the declared grammar")
    train_games = {row["source_game_id"] for row in stage_a if row["record_type"].startswith("v7-")}
    rollout_games = {row["source_game_id"] for row in rollout}
    gate_games = {row["source_game_id"] for row in gate}
    if train_games & rollout_games or (train_games | rollout_games) & gate_games:
        raise RuntimeError("Whole-game split leakage in packaged V7 data")
    result = {
        "audit": "passed",
        "source_sha256": sha256(ARCHIVE),
        "stage_a_records": len(stage_a),
        "validation_records": len(validation),
        "rollout_records": len(rollout),
        "gate_records": len(gate),
        "gate_games": len(gate_games),
        "record_types": dict(sorted(Counter(row["record_type"] for row in stage_a).items())),
        "free_plan_grammar_targets": len(free_targets),
        "split_overlap": 0,
        "official_positions_used": data_manifest["contamination"]["official_positions_used"],
        "token_audit": data_manifest["token_audit"],
    }
    Path("/content/v7_colab_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
