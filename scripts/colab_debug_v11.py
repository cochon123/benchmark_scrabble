from __future__ import annotations

import os
import subprocess
import sys


environment = os.environ.copy()
environment["PYTHONPATH"] = "/content/v11"
environment["HF_HOME"] = "/content/hf_cache"
command = [
    sys.executable,
    "/content/v11/scripts/evaluate_v11_policy.py",
    "--model",
    "Qwen/Qwen3-4B-Instruct-2507",
    "--dataset",
    "/content/v11/data/v11_atomic_policy/frozen_gate.json",
    "--output",
    "/content/v11/artifacts/v11_colab/evaluations/debug.json",
    "--categories",
    "opening",
    "--limit-per-category",
    "1",
]
result = subprocess.run(command, env=environment, text=True, capture_output=True)
print("RETURN", result.returncode)
print("STDOUT\n", result.stdout[-12000:])
print("STDERR\n", result.stderr[-12000:])
