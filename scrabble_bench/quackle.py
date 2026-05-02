from __future__ import annotations

import subprocess
from pathlib import Path

from .config import QUACKLE_DIR, ensure_directories


QUACKLE_REPO = "https://github.com/quackle/quackle.git"


def setup_quackle() -> None:
    ensure_directories()
    checkout_dir = QUACKLE_DIR / "src"
    if not checkout_dir.exists():
        subprocess.run(["git", "clone", QUACKLE_REPO, str(checkout_dir)], check=True)
    build_dir = checkout_dir / "quacker" / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
    subprocess.run(["cmake", "--build", "."], cwd=build_dir, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout_dir, text=True).strip()
    (QUACKLE_DIR / "BUILD_INFO.txt").write_text(
        f"repo={QUACKLE_REPO}\ncommit={commit}\nbinary={build_dir / 'Quackle'}\n",
        encoding="utf-8",
    )

