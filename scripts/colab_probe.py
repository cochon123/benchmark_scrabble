#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import shutil
import subprocess


def main() -> None:
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb": (
                round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
                if torch.cuda.is_available()
                else None
            ),
            "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        }
    except Exception as exc:
        torch_info = {"error": str(exc)}
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "disk": shutil.disk_usage("/content")._asdict(),
                "torch": torch_info,
            },
            indent=2,
        )
    )
    subprocess.run(["nvidia-smi"], check=False)


if __name__ == "__main__":
    main()
