#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def find_shards(root: Path) -> list[Path]:
    shards = {
        path.parent.resolve()
        for path in root.rglob("manifest.json")
        if (path.parent / "train_positions.json").exists()
        and path.parent.name.startswith("shard-")
    }
    return sorted(shards)


def run(command: list[str], root: Path) -> None:
    print("[pipeline] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge, audit, and render downloaded v3 position shards."
    )
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=12)
    parser.add_argument("--positions-dir", type=Path, default=Path("data/general_v3_merged"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/general_v3_sft"))
    parser.add_argument("--candidate-limit", type=int, default=3)
    parser.add_argument("--audit-rate", type=float, default=0.5)
    parser.add_argument("--transpose-rate", type=float, default=0.5)
    parser.add_argument("--hard-negative-sft", type=Path, action="append", default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    shards = find_shards(args.shards_root)
    if len(shards) != args.expected_shards:
        raise RuntimeError(
            f"Expected {args.expected_shards} complete shards, found {len(shards)}: {shards}"
        )
    merge = [
        sys.executable,
        "scripts/merge_general_position_shards.py",
        *[str(path) for path in shards],
        "--output-dir",
        str(args.positions_dir),
    ]
    run(merge, root)
    run(
        [
            sys.executable,
            "scripts/audit_general_dataset.py",
            str(args.positions_dir),
        ],
        root,
    )
    build = [
        sys.executable,
        "scripts/build_general_v3_data.py",
        "--positions-dir",
        str(args.positions_dir),
        "--output-dir",
        str(args.output_dir),
        "--candidate-limit",
        str(args.candidate_limit),
        "--audit-rate",
        str(args.audit_rate),
        "--transpose-rate",
        str(args.transpose_rate),
    ]
    for path in args.hard_negative_sft:
        build.extend(["--hard-negative-sft", str(path)])
    run(build, root)
    state = {
        "state": "ready-for-packaging",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "shards": [str(path) for path in shards],
        "positions_dir": str(args.positions_dir),
        "sft_dir": str(args.output_dir),
        "dataset_audit": json.loads(
            (args.positions_dir / "audit.json").read_text(encoding="utf-8")
        ),
        "sft_manifest": json.loads(
            (args.output_dir / "manifest.json").read_text(encoding="utf-8")
        ),
    }
    (args.output_dir / "pipeline_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
