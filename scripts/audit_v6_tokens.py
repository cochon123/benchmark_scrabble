#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def token_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/general_v6_pilot"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    report: dict[str, Any] = {"model": args.model, "thinking_disabled": True, "variants": {}}
    for variant in ("direct", "action_value", "compact_search"):
        lengths: list[int] = []
        completion_lengths: list[int] = []
        for split in ("train", "validation"):
            path = args.data_dir / f"{variant}_{split}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                lengths.append(len(token_ids(tokenizer, row["messages"])))
                completion_lengths.append(
                    len(tokenizer(row["messages"][-1]["content"], add_special_tokens=False)["input_ids"])
                )
        report["variants"][variant] = {
            "records": len(lengths),
            "p50": percentile(lengths, 0.5),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths),
            "completion_mean": sum(completion_lengths) / len(completion_lengths),
            "completion_p99": percentile(completion_lengths, 0.99),
            "total_tokens": sum(lengths),
        }
    maximum = max(item["max"] for item in report["variants"].values())
    report["recommended_max_length"] = ((maximum + 63) // 64) * 64
    output = args.data_dir / "token_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
