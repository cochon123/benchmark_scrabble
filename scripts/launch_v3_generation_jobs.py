#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Print or launch independent paid HF v3 generation jobs.")
    parser.add_argument("--input-repo", required=True)
    parser.add_argument(
        "--output-repo-prefix",
        required=True,
        help="Each shard uses PREFIX-00, PREFIX-01, etc. to avoid concurrent Hub commits.",
    )
    parser.add_argument("--seed-start", type=int, default=500_000)
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--games-per-shard", type=int, default=100)
    parser.add_argument("--flavor", default="cpu-performance")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    jobs = []
    for shard in range(args.shards):
        seed = args.seed_start + shard * args.games_per_shard
        output_repo = f"{args.output_repo_prefix}-{shard:02d}"
        command = [
            "hf", "jobs", "uv", "run",
            "--flavor", args.flavor,
            "--secrets", "HF_TOKEN",
            "--volume", f"hf://datasets/{args.input_repo}:/input:ro",
            "--label", "project=scrabble-v3-data",
            "--label", f"shard={shard:02d}",
            "--detach",
            "scripts/hf_generate_v3_shard_job.py",
            "--shard-index", str(shard),
            "--games", str(args.games_per_shard),
            "--seed-start", str(seed),
            "--output-repo", output_repo,
        ]
        row = {
            "shard": shard,
            "seed_start": seed,
            "output_repo": output_repo,
            "command": command,
        }
        if args.launch:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            row["job_output"] = result.stdout.strip()
        jobs.append(row)
    print(json.dumps({"launched": args.launch, "jobs": jobs}, indent=2))


if __name__ == "__main__":
    main()
