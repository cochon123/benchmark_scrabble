#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys


os.chdir("/content/scrabble_ai")
sys.argv = [
    "scripts/evaluate_hf.py",
    "--boards",
    "5",
    "--output",
    "/content/baseline-smoke.json",
]
_ = runpy.run_path("scripts/evaluate_hf.py", run_name="__main__")
