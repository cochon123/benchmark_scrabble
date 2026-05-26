from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DB_DIR = REPO_ROOT / "db"
DB_PATH = DB_DIR / "benchmark.sqlite"
LOG_DIR = DATA_DIR / "logs"
EXPORT_DIR = DATA_DIR / "exports"
DATASET_DIR = DATA_DIR / "dataset"
DATASET_PATH = DATASET_DIR / "benchmark_positions.json"
LEXICON_DIR = DATA_DIR / "lexicon"
PRIMARY_LEXICON_PATH = LEXICON_DIR / "NWL23.txt"
SECONDARY_LEXICON_PATHS = [
    LEXICON_DIR / "ENABLE.txt",
]
FALLBACK_LEXICON_PATHS = [
    Path("/usr/share/dict/american-english"),
    Path("/usr/share/dict/words"),
]
QUACKLE_DIR = DATA_DIR / "quackle"
ENV_PATH = REPO_ROOT / ".env"


def ensure_directories() -> None:
    for path in (DATA_DIR, DB_DIR, LOG_DIR, EXPORT_DIR, DATASET_DIR, LEXICON_DIR, QUACKLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_env_file() -> None:
    if not ENV_PATH.exists():
        return
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def get_openrouter_api_key() -> str | None:
    load_env_file()
    return os.environ.get("OPENROUTER_API_KEY")


VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def normalize_reasoning_effort(effort: str | None) -> str:
    value = (effort or "medium").strip().lower()
    if value not in VALID_REASONING_EFFORTS:
        raise RuntimeError(
            "Reasoning effort must be one of: "
            + ", ".join(sorted(VALID_REASONING_EFFORTS))
        )
    return value


def get_openrouter_reasoning_effort() -> str:
    load_env_file()
    effort = os.environ.get("OPENROUTER_REASONING_EFFORT", "medium")
    return normalize_reasoning_effort(effort)


def resolve_lexicon_path() -> Path:
    if PRIMARY_LEXICON_PATH.exists():
        return PRIMARY_LEXICON_PATH
    for path in SECONDARY_LEXICON_PATHS:
        if path.exists():
            return path
    for path in FALLBACK_LEXICON_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No lexicon file found. Add data/lexicon/NWL23.txt or install a system word list."
    )
