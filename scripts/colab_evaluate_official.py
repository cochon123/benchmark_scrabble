#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys


os.chdir("/content/scrabble_ai")
sys.argv = [
    "scripts/evaluate_hf.py",
    "--adapter",
    "/content/qwen3-4b-scrabble-clean-lora",
    "--output",
    "/content/official-full-evaluation.json",
]
_ = runpy.run_path("scripts/evaluate_hf.py", run_name="__main__")

