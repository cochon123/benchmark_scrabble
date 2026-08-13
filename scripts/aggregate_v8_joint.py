#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for objective in ("canonical", "set-valued"):
        for seed in (8101, 8102, 8103):
            metrics = json.loads((args.directory / f"{objective}-{seed}.metrics.json").read_text())
            plain_name = (
                f"{objective}-{seed}-balanced-evaluation.json"
                if objective == "canonical" and seed == 8101
                else f"{objective}-{seed}-evaluation.json"
            )
            plain = json.loads((args.directory / plain_name).read_text())["summary"]
            constrained = json.loads(
                (args.directory / f"{objective}-{seed}-constrained-evaluation.json").read_text()
            )["summary"]
            runs.append({
                "objective": objective, "seed": seed,
                "validation_loss": metrics["validation_loss"],
                "train_elapsed_seconds": metrics["elapsed_seconds"],
                "history": metrics["history"], "plain": plain, "lexicon_constrained": constrained,
            })
    aggregates = {}
    for objective in ("canonical", "set-valued"):
        subset = [run for run in runs if run["objective"] == objective]
        aggregates[objective] = {}
        for mode in ("plain", "lexicon_constrained"):
            aggregates[objective][mode] = {
                metric: mean_sd([run[mode][metric] for run in subset])
                for metric in (
                    "top1_legal_pct", "top1_optimal_pct", "top1_score_pct",
                    "pass32_legal_pct", "pass32_optimal_pct",
                )
            }
        aggregates[objective]["validation_loss"] = mean_sd(
            [run["validation_loss"] for run in subset]
        )
        aggregates[objective]["train_elapsed_seconds"] = mean_sd(
            [run["train_elapsed_seconds"] for run in subset]
        )
    payload = {"aggregates": aggregates, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
