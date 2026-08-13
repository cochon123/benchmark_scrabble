#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys


os.chdir("/content/scrabble_ai")
sys.argv = [
    "scripts/generate_training_data.py",
    "--games",
    "128",
    "--workers",
    "2",
    "--seed-start",
    "300000",
    "--output-dir",
    "/content/scrabble_ai/data/general_v2_remote",
]
_ = runpy.run_path("scripts/generate_training_data.py", run_name="__main__")
