# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=0.27"]
# ///
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download all completed v3 HF position shards.")
    parser.add_argument("--repo-prefix", required=True)
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("data/general_v3_shards"))
    args = parser.parse_args()
    downloads = []
    for shard in range(args.shards):
        repo = f"{args.repo_prefix}-{shard:02d}"
        destination = args.output_dir / f"repo-{shard:02d}"
        path = snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=destination,
        )
        downloads.append({"shard": shard, "repo": repo, "path": path})
    print(json.dumps({"downloads": downloads}, indent=2))


if __name__ == "__main__":
    main()
