#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys


packages = [
    "accelerate>=1.2",
    "bitsandbytes>=0.45",
    "datasets>=3.2",
    "peft>=0.14",
    "safetensors>=0.4",
    "transformers>=4.55",
]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", *packages],
    check=True,
)
print("Installed general-training dependencies")
