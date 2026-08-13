#!/usr/bin/env python3
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data/general_v5_8b"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", use_fast=True)
    lengths: list[int] = []
    for name in ("train.jsonl", "validation.jsonl"):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            tokens = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=False, enable_thinking=False
            )
            if isinstance(tokens, Mapping):
                tokens = tokens["input_ids"]
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
            if tokens and isinstance(tokens[0], list):
                tokens = tokens[0]
            lengths.append(len(tokens))
    lengths.sort()

    def percentile(value: float) -> int:
        return lengths[min(len(lengths) - 1, round((len(lengths) - 1) * value))]

    result = {
        "records": len(lengths),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(lengths),
        "thinking_disabled": True,
    }
    (root / "token_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
