#!/usr/bin/env python3
"""Terminate stale GPU compute processes while preserving this Colab kernel."""

from __future__ import annotations

import os
import signal
import subprocess
import time


def gpu_processes() -> list[tuple[int, int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        pid_text, memory_text = (part.strip() for part in line.split(",", 1))
        processes.append((int(pid_text), int(memory_text)))
    return processes


current_pid = os.getpid()
stale = [(pid, mib) for pid, mib in gpu_processes() if pid != current_pid]
print(f"Current kernel PID: {current_pid}")
print(f"Stale GPU processes: {stale}")
for pid, _ in stale:
    os.kill(pid, signal.SIGTERM)

if stale:
    time.sleep(5)
print(f"Remaining GPU processes: {gpu_processes()}")
