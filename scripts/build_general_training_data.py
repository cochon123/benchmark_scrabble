#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.training import recovery_training_record, training_record


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> tuple[str, int]:
    payload = "".join(
        json.dumps(record, separators=(",", ":")) + "\n" for record in records
    )
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(payload.encode("utf-8"))


def error_category(error: str) -> str:
    if "Rack cannot supply letter" in error:
        return "rack"
    if "No new tiles were provided" in error:
        return "empty"
    if "collides with existing tile" in error:
        return "collision"
    if "same row or column" in error or "gap in its main word" in error:
        return "geometry"
    if "invalid" in error.lower():
        return "word"
    if "does not connect" in error:
        return "connection"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/general_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/general_v2_sft"))
    parser.add_argument(
        "--recovery-rate",
        type=float,
        default=0.5,
        help="Deterministic fraction of training positions that get a retry-recovery record.",
    )
    args = parser.parse_args()
    if not 0 <= args.recovery_rate <= 1:
        raise SystemExit("--recovery-rate must be between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    manifest: dict[str, Any] = {
        "source": str(args.input_dir),
        "benchmark_contaminated": False,
        "include_reasoning": True,
        "recovery_rate": args.recovery_rate,
        "splits": {},
    }

    for split in ("train", "validation", "test"):
        positions_path = args.input_dir / f"{split}_positions.json"
        positions = json.loads(positions_path.read_text(encoding="utf-8"))
        records = []
        recovery_count = 0
        recovery_errors: Counter[str] = Counter()
        for position in positions:
            transform = (
                "transpose"
                if int(hashlib.sha256(position["id"].encode()).hexdigest()[:8], 16) % 2
                else "identity"
            )
            records.append(
                {
                    **training_record(
                        position,
                        transform,
                        include_reasoning=True,
                    ),
                    "record_type": "optimal-move",
                }
            )
            selector = int(hashlib.sha256((position["id"] + "retry").encode()).hexdigest()[:8], 16)
            if split == "train" and selector / 0xFFFFFFFF < args.recovery_rate:
                recovery = recovery_training_record(
                    position,
                    lexicon,
                    transform,
                    include_reasoning=True,
                )
                records.append(recovery)
                recovery_count += 1
                recovery_errors[error_category(recovery["rejection_error"])] += 1
        jsonl_sha256, jsonl_bytes = write_jsonl(
            args.output_dir / f"{split}.jsonl", records
        )
        positions_payload = json.dumps(positions, indent=2)
        (args.output_dir / f"{split}_positions.json").write_text(positions_payload, encoding="utf-8")
        manifest["splits"][split] = {
            "positions": len(positions),
            "records": len(records),
            "recovery_records": recovery_count,
            "recovery_error_categories": dict(sorted(recovery_errors.items())),
            "jsonl_sha256": jsonl_sha256,
            "jsonl_bytes": jsonl_bytes,
            "positions_sha256": hashlib.sha256(positions_payload.encode("utf-8")).hexdigest(),
        }

    selection_path = args.input_dir / "selection_positions.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        (args.output_dir / "selection_positions.json").write_text(
            json.dumps(selection, indent=2),
            encoding="utf-8",
        )
        manifest["selection_positions"] = len(selection)
        manifest["selection_sha256"] = hashlib.sha256(
            json.dumps(selection, indent=2).encode("utf-8")
        ).hexdigest()

    audit_path = args.input_dir / "audit.json"
    if audit_path.exists():
        shutil.copy2(audit_path, args.output_dir / "source_audit.json")
        manifest["source_audit"] = "source_audit.json"

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
