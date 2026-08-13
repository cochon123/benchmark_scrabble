#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys


os.chdir("/content/scrabble_ai")
sys.argv = [
    "scripts/train_qlora.py",
    "--output-dir",
    "/content/scrabble-train-smoke",
    "--max-train-samples",
    "8",
    "--max-validation-samples",
    "4",
    "--max-steps",
    "1",
]
_ = runpy.run_path("scripts/train_qlora.py", run_name="__main__")

