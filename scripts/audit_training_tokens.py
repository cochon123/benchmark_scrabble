#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def distribution(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "p99": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure exact chat-template lengths before training.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    rows: list[dict[str, Any]] = []
    for line in args.data.open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        encoded = tokenizer.apply_chat_template(
            record["messages"][:-1],
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        prompt_length = len(encoded)
        completion_length = len(
            tokenizer(record["messages"][-1]["content"], add_special_tokens=False)[
                "input_ids"
            ]
        ) + 1
        rows.append(
            {
                "id": record.get("id"),
                "record_type": record.get("record_type", "unknown"),
                "prompt": prompt_length,
                "completion": completion_length,
                "total": prompt_length + completion_length,
            }
        )
    if not rows:
        raise RuntimeError(f"No records in {args.data}")
    over_limit = [row for row in rows if int(row["total"]) > args.max_length]
    types = Counter(str(row["record_type"]) for row in rows)
    report = {
        "model": args.model,
        "data": str(args.data),
        "records": len(rows),
        "max_length": args.max_length,
        "prompt": distribution([int(row["prompt"]) for row in rows]),
        "completion": distribution([int(row["completion"]) for row in rows]),
        "total": distribution([int(row["total"]) for row in rows]),
        "record_types": dict(sorted(types.items())),
        "over_limit": len(over_limit),
        "longest": sorted(rows, key=lambda row: int(row["total"]), reverse=True)[:10],
        "passed": not over_limit,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if over_limit:
        raise SystemExit(f"{len(over_limit)} records exceed max length {args.max_length}")


if __name__ == "__main__":
    main()
