#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys


os.chdir("/content/scrabble_ai")
smoke_file = "/content/scrabble_ai/data/training/benchmark_smoke_train.jsonl"
sys.argv = [
    "scripts/train_qlora.py",
    "--train-file",
    smoke_file,
    "--validation-file",
    smoke_file,
    "--init-adapter",
    "/content/qwen3-4b-scrabble-specialized-lora/checkpoint-150",
    "--output-dir",
    "/content/qwen3-4b-scrabble-smoke-specialized-lora",
    "--epochs",
    "20",
    "--learning-rate",
    "0.0002",
    "--gradient-accumulation",
    "2",
    "--logging-steps",
    "5",
    "--save-total-limit",
    "2",
]
_ = runpy.run_path("scripts/train_qlora.py", run_name="__main__")
