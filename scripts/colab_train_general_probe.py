#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys


os.chdir("/content/scrabble_ai")
sys.argv = [
    "scripts/train_qlora.py",
    "--model",
    "Qwen/Qwen3-4B-Thinking-2507",
    "--train-file",
    "/content/scrabble_ai/data/general_v2_sft/train.jsonl",
    "--validation-file",
    "/content/scrabble_ai/data/general_v2_sft/validation.jsonl",
    "--output-dir",
    "/content/qwen3-4b-scrabble-general-probe",
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
    "1",
    "--max-train-samples",
    "32",
    "--max-validation-samples",
    "8",
    "--max-steps",
    "5",
    "--logging-steps",
    "1",
    "--save-total-limit",
    "1",
]
_ = runpy.run_path("scripts/train_qlora.py", run_name="__main__")
