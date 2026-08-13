#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.v3_training import v3_audit_record, v3_training_record


def selected(identifier: str, rate: float, suffix: str) -> bool:
    value = int(hashlib.sha256(f"{identifier}:{suffix}".encode()).hexdigest()[:8], 16)
    return value / 0xFFFFFFFF < rate


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> tuple[str, int]:
    payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest(), len(payload.encode())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hard_negative_rows(
    rows: list[dict[str, Any]], authorized_source_ids: set[str]
) -> None:
    unauthorized = sorted(
        {
            str(row.get("source_id"))
            for row in rows
            if str(row.get("source_id")) not in authorized_source_ids
        }
    )
    if unauthorized:
        raise RuntimeError(
            "Hard-negative SFT contains non-train source IDs: "
            f"{unauthorized[:10]}"
        )
    malformed = [
        str(row.get("id"))
        for row in rows
        if not isinstance(row.get("messages"), list)
        or not row["messages"]
        or not isinstance(row["messages"][-1], dict)
        or row["messages"][-1].get("role") != "assistant"
    ]
    if malformed:
        raise RuntimeError(
            "Hard-negative SFT contains malformed conversations: "
            f"{malformed[:10]}"
        )


def normalized_hard_negative_row(
    row: dict[str, Any], positions_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    position = positions_by_id[str(row["source_id"])]
    return {
        "id": str(row["id"]),
        "source_id": str(row["source_id"]),
        "transform": "identity",
        "position_key": row.get("position_key", position.get("position_key")),
        "optimal_score": int(position["optimal_score"]),
        "record_type": str(row["record_type"]),
        "messages": row["messages"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build process-supervised Scrabble v3 data.")
    parser.add_argument("--positions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/general_v3_sft"))
    parser.add_argument("--candidate-limit", type=int, default=3)
    parser.add_argument("--audit-rate", type=float, default=0.5)
    parser.add_argument("--transpose-rate", type=float, default=0.5)
    parser.add_argument("--hard-negative-sft", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not 0 <= args.audit_rate <= 1 or not 0 <= args.transpose_rate <= 1:
        raise SystemExit("Rates must be between zero and one")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    manifest: dict[str, Any] = {
        "version": 3,
        "positions_dir": str(args.positions_dir),
        "board_encoding": "dense-grid",
        "process_supervision": True,
        "candidate_limit": args.candidate_limit,
        "audit_rate": args.audit_rate,
        "transpose_rate": args.transpose_rate,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        path = args.positions_dir / f"{split}_positions.json"
        positions = json.loads(path.read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = []
        for position in positions:
            transform = (
                "transpose"
                if split == "train"
                and selected(str(position["id"]), args.transpose_rate, "transpose")
                else "identity"
            )
            rows.append(
                v3_training_record(
                    position,
                    lexicon,
                    transform,
                    candidate_limit=args.candidate_limit,
                )
            )
            if split == "train" and selected(
                str(position["id"]),
                args.audit_rate,
                "audit",
            ):
                rows.append(v3_audit_record(position, lexicon, transform))
        hard_negative_count = 0
        if split == "train":
            positions_by_id = {str(position["id"]): position for position in positions}
            authorized_source_ids = set(positions_by_id)
            for hard_negative_path in args.hard_negative_sft:
                extra = read_jsonl(hard_negative_path)
                validate_hard_negative_rows(extra, authorized_source_ids)
                rows.extend(
                    normalized_hard_negative_row(row, positions_by_id) for row in extra
                )
                hard_negative_count += len(extra)
        record_ids = [str(row["id"]) for row in rows]
        if len(record_ids) != len(set(record_ids)):
            raise RuntimeError(f"Duplicate SFT record IDs in {split} split")
        digest, size = write_jsonl(args.output_dir / f"{split}.jsonl", rows)
        manifest["splits"][split] = {
            "positions": len(positions),
            "positions_path": str(path),
            "positions_sha256": file_sha256(path),
            "records": len(rows),
            "hard_negative_records": hard_negative_count,
            "sha256": digest,
            "bytes": size,
        }
        shutil.copy2(path, args.output_dir / path.name)
    for name in (
        "selection_positions.json",
        "train_mining_positions.json",
        "audit.json",
        "manifest.json",
    ):
        source = args.positions_dir / name
        if source.exists():
            destination_name = {
                "selection_positions.json": "selection_positions.json",
                "train_mining_positions.json": "train_mining_positions.json",
                "audit.json": "source_audit.json",
                "manifest.json": "source_positions_manifest.json",
            }[name]
            shutil.copy2(source, args.output_dir / destination_name)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
