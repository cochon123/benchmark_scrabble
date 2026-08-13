#!/usr/bin/env python3
"""Package the frozen V11 true-cross MML run for Hugging Face Jobs."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from huggingface_hub import HfApi


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive(output: Path, root: Path, paths: list[Path]) -> None:
    with tarfile.open(output, "w:gz") as target:
        for path in paths:
            target.add(path, arcname=str(path.relative_to(root)))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/v11_true_cross_mml_hf_bundle"
    output.mkdir(parents=True, exist_ok=True)
    paths = [root / "pyproject.toml", root / "data/lexicon/ENABLE.txt"]
    paths += list((root / "scrabble_bench").rglob("*.py"))
    paths += [root / "scripts/train_qlora.py", root / "scripts/evaluate_v11_policy.py"]
    paths += list((root / "data/v11_true_cross_mml").glob("*.jsonl"))
    paths += [root / "data/v11_true_cross_mml/gate.json"]
    source = output / "source.tar.gz"
    archive(source, root, paths)
    adapter = root / "artifacts/v11_atomic_policy/colab/stage1d_anchor_step_25.tar.gz"
    staged_adapter = output / "stage1d_adapter.tar.gz"
    staged_adapter.write_bytes(adapter.read_bytes())
    manifest = {
        "version": "v11-true-cross-exact-mml-1",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "output_repo": "Cochon123/Qwen3-4B-Scrabble-V11-True-Cross-MML",
        "hardware": "a100-large",
        "cost_per_hour_usd": 2.5,
        "timeout": "100m",
        "max_cost_usd": 4.167,
        "max_steps": 125,
        "learning_rate": 1e-5,
        "seed": 13021,
        "input_sha256": {
            "source.tar.gz": sha256(source),
            "stage1d_adapter.tar.gz": sha256(staged_adapter),
        },
        "frozen_gate_sha256": sha256(root / "data/v11_true_cross_mml/gate.json"),
    }
    (output / "job_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "hf_train_v11_true_cross_mml_job.py").write_bytes(
        (root / "scripts/hf_train_v11_true_cross_mml_job.py").read_bytes()
    )
    repo = "Cochon123/scrabble-v11-true-cross-mml-input"
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(output), repo_id=repo, repo_type="dataset", commit_message="Upload V11 true-cross exact-MML job bundle")
    print(json.dumps({**manifest, "input_repo": repo}, indent=2))


if __name__ == "__main__":
    main()
