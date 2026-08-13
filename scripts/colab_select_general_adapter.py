#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


root = Path("/content/qwen3-4b-scrabble-general-lora")
destination = Path("/content/qwen3-4b-scrabble-general-selected")
candidates = [
    {
        "name": "checkpoint-250",
        "adapter": root / "checkpoint-250",
        "evaluation": Path("/content/general-selection-checkpoint250-evaluation.json"),
    },
    {
        "name": "checkpoint-1000",
        "adapter": root / "checkpoint-1000",
        "evaluation": Path("/content/general-selection-checkpoint1000-evaluation.json"),
    },
    {
        "name": "final-step-1265",
        "adapter": root,
        "evaluation": Path("/content/general-selection-evaluation.json"),
    },
]


def selection_key(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    regret = summary.get("mean_regret_given_legal")
    return (
        float(summary["score_pct"]),
        float(summary["optimal_move_pct"]),
        float(summary["legal_move_pct"]),
        -float(regret) if regret is not None else float("-inf"),
    )


for candidate in candidates:
    payload = json.loads(candidate["evaluation"].read_text(encoding="utf-8"))
    candidate["summary"] = payload["summary"]

chosen = max(candidates, key=lambda item: selection_key(item["summary"]))
if destination.exists():
    shutil.rmtree(destination)
destination.mkdir(parents=True)

# Root-level tokenizer/configuration files are shared by all checkpoints.
for path in root.iterdir():
    if path.is_file():
        shutil.copy2(path, destination / path.name)
# A checkpoint choice replaces the final adapter weights/configuration.
if chosen["adapter"] != root:
    for name in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
        source = chosen["adapter"] / name
        if source.exists():
            shutil.copy2(source, destination / name)

manifest = json.loads(
    Path("/content/scrabble_ai/data/general_v2_sft/manifest.json").read_text(
        encoding="utf-8"
    )
)
record = {
    "rule": [
        "highest validation score_pct",
        "highest optimal_move_pct",
        "highest legal_move_pct",
        "lowest mean_regret_given_legal",
    ],
    "selection_dataset": "/content/scrabble_ai/data/general_v2_sft/selection_positions.json",
    "selection_positions": manifest["selection_positions"],
    "selection_sha256": manifest["selection_sha256"],
    "candidates": [
        {
            "name": item["name"],
            "adapter": str(item["adapter"]),
            "evaluation": str(item["evaluation"]),
            "summary": item["summary"],
        }
        for item in candidates
    ],
    "chosen": chosen["name"],
}
selection_path = Path("/content/general-checkpoint-selection.json")
selection_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
shutil.copy2(selection_path, destination / "checkpoint_selection.json")
print(json.dumps(record, indent=2))
