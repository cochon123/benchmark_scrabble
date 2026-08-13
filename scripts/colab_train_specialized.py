#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys


os.chdir("/content/scrabble_ai")
sys.argv = [
    "scripts/train_qlora.py",
    "--train-file",
    "/content/scrabble_ai/data/training/benchmark_specialized_train.jsonl",
    "--validation-file",
    "/content/scrabble_ai/data/training/validation.jsonl",
    "--init-adapter",
    "/content/qwen3-4b-scrabble-clean-lora",
    "--output-dir",
    "/content/qwen3-4b-scrabble-specialized-lora",
    "--epochs",
    "3",
    "--learning-rate",
    "0.0001",
    "--gradient-accumulation",
    "4",
]
_ = runpy.run_path("scripts/train_qlora.py", run_name="__main__")
