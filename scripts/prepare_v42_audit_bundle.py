#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from pathlib import Path


def archive(output: Path, entries: list[tuple[Path, str]]) -> None:
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as target:
                for path, name in sorted(entries, key=lambda item: item[1]):
                    info = target.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        target.addfile(info, handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "v42_audit_bundle"
    output.mkdir(parents=True, exist_ok=True)
    source = [(root / "pyproject.toml", "pyproject.toml"), (root / "data/lexicon/ENABLE.txt", "data/lexicon/ENABLE.txt")]
    source += [(path, str(path.relative_to(root))) for path in (root / "scrabble_bench").rglob("*.py")]
    source += [(root / "scripts/verify_v42_data.py", "scripts/verify_v42_data.py")]
    source += [
        (root / "data/general_v4_pilot/train_positions.json", "data/general_v4_pilot/train_positions.json"),
        (root / "data/general_v3_merged/test_positions.json", "data/general_v3_merged/test_positions.json"),
        (root / "data/dataset/benchmark_positions.json", "data/dataset/benchmark_positions.json"),
    ]
    data_root = root / "data/general_v42"
    archive(output / "v42_source.tar.gz", source)
    archive(output / "v42_data.tar.gz", [(path, str(path.relative_to(root))) for path in data_root.rglob("*") if path.is_file()])
    payload = {name: sha256(output / name) for name in ("v42_source.tar.gz", "v42_data.tar.gz")}
    (output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
