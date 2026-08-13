#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys


os.chdir("/content/scrabble_ai")
sys.argv = [
    "scripts/evaluate_hf.py",
    "--model",
    "Qwen/Qwen3-4B-Thinking-2507",
    "--adapter",
    "/content/qwen3-4b-scrabble-general-lora",
    "--dataset",
    "/content/scrabble_ai/data/general_v2_sft/selection_positions.json",
    "--max-new-tokens",
    "256",
    "--batch-size",
    "2",
    "--do-sample",
    "--output",
    "/content/general-selection-evaluation.json",
]
_ = runpy.run_path("scripts/evaluate_hf.py", run_name="__main__")
