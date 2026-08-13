#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path


def main() -> None:
    archive = Path("/content/scrabble_ai_source.tar.gz")
    destination = Path("/content/scrabble_ai")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(destination, filter="data")
    subprocess.run(
        ["python", "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(destination)],
        check=True,
    )
    print(f"Extracted source to {destination}")


if __name__ == "__main__":
    main()
