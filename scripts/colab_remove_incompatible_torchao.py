#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys


subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
print("Removed optional incompatible torchao package; restart the kernel before PEFT import")
