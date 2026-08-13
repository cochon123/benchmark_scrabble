#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys


os.chdir("/content/scrabble_ai")
sys.argv = [
    "scripts/train_qlora.py",
    "--output-dir",
    "/content/qwen3-4b-scrabble-clean-lora",
    "--epochs",
    "1",
    "--learning-rate",
    "0.0002",
    "--gradient-accumulation",
    "8",
]
_ = runpy.run_path("scripts/train_qlora.py", run_name="__main__")
