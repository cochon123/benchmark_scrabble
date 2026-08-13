#!/usr/bin/env python3
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507", use_fast=True)
lengths = []


def token_ids(messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


for line in Path("/content/scrabble_ai/data/general_v2_sft/train.jsonl").read_text(encoding="utf-8").splitlines():
    record = json.loads(line)
    prompt_ids = token_ids(record["messages"][:-1], add_generation_prompt=True)
    completion_ids = tokenizer(
        record["messages"][-1]["content"],
        add_special_tokens=False,
    )["input_ids"]
    lengths.append(
        {
            "id": record["id"],
            "record_type": record.get("record_type", "optimal-move"),
            "prompt": len(prompt_ids),
            "completion": len(completion_ids) + 1,
            "total": len(prompt_ids) + len(completion_ids) + 1,
        }
    )


def distribution(field: str, rows: list[dict[str, object]]) -> dict[str, int]:
    values = sorted(int(item[field]) for item in rows)
    return {
        "min": values[0],
        "median": values[len(values) // 2],
        "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
        "p99": values[min(len(values) - 1, int(len(values) * 0.99))],
        "max": values[-1],
    }


print(
    json.dumps(
        {
            "count": len(lengths),
            "prompt": distribution("prompt", lengths),
            "completion": distribution("completion", lengths),
            "total": distribution("total", lengths),
            "over_2048": sum(int(item["total"]) > 2048 for item in lengths),
            "over_2560": sum(int(item["total"]) > 2560 for item in lengths),
            "by_record_type": {
                kind: {
                    "count": len(rows),
                    "total": distribution("total", rows),
                }
                for kind in sorted({str(item["record_type"]) for item in lengths})
                for rows in [[item for item in lengths if item["record_type"] == kind]]
            },
            "first": lengths[:8],
        },
        indent=2,
    )
)
