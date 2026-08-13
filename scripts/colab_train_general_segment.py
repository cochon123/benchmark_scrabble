#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path


os.chdir("/content/scrabble_ai")
output = Path("/content/qwen3-4b-scrabble-general-lora")
checkpoints = sorted(
    output.glob("checkpoint-*"),
    key=lambda path: int(path.name.rsplit("-", 1)[1]),
)
resume = checkpoints[-1] if checkpoints else None
current_step = 0
if resume:
    state = json.loads((resume / "trainer_state.json").read_text(encoding="utf-8"))
    current_step = int(state["global_step"])
stop_after = min(current_step + 100, 1265)

sys.argv = [
    "scripts/train_qlora.py",
    "--model",
    "Qwen/Qwen3-4B-Thinking-2507",
    "--train-file",
    "/content/scrabble_ai/data/general_v2_sft/train.jsonl",
    "--validation-file",
    "/content/scrabble_ai/data/general_v2_sft/validation.jsonl",
    "--output-dir",
    str(output),
    "--max-length",
    "2816",
    "--lora-r",
    "16",
    "--lora-alpha",
    "32",
    "--epochs",
    "1",
    "--learning-rate",
    "0.0001",
    "--gradient-accumulation",
    "4",
    "--logging-steps",
    "25",
    "--checkpoint-every",
    "50",
    "--save-total-limit",
    "30",
    "--stop-after-step",
    str(stop_after),
]
if resume:
    sys.argv.extend(["--resume-from-checkpoint", str(resume)])
if stop_after < 1265:
    sys.argv.append("--skip-final-eval")
print(
    json.dumps(
        {
            "resume": str(resume) if resume else None,
            "current_step": current_step,
            "stop_after_step": stop_after,
        },
        indent=2,
    )
)
_ = runpy.run_path("scripts/train_qlora.py", run_name="__main__")
