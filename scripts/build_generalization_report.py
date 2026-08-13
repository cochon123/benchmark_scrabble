#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Any

from build_experiment_report import COLORS, grouped_bar_chart, polyline_chart


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return 100 * (center - half), 100 * (center + half)


def score_interval(results: list[dict[str, Any]], seed: int = 3407) -> tuple[float, float]:
    if not results:
        return 0.0, 0.0
    clusters: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        match = re.match(r"train-(\d+)-", str(result["id"]))
        key = match.group(1) if match else str(result["id"])
        clusters.setdefault(key, []).append(result)
    groups = list(clusters.values())
    rng = random.Random(seed)
    values = []
    for _ in range(10_000):
        sample = [rng.choice(groups) for _ in groups]
        raw = sum(int(item["score"]) for group in sample for item in group)
        optimal = sum(int(item["optimal_score"]) for group in sample for item in group)
        values.append(100 * raw / optimal if optimal else 0.0)
    values.sort()
    return values[249], values[9749]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the leakage-safe generalization report.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/generalization_experiment.json")
    )
    parser.add_argument(
        "--dataset-manifest", type=Path, default=Path("data/general_v2_sft/manifest.json")
    )
    parser.add_argument(
        "--audit", type=Path, default=Path("data/general_v2_sft/source_audit.json")
    )
    parser.add_argument(
        "--dataset-statistics",
        type=Path,
        default=Path("data/general_v2_sft/dataset_statistics.json"),
    )
    parser.add_argument("--trainer-state", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--selection-record", type=Path, required=True)
    parser.add_argument(
        "--evaluation", action="append", default=[], type=named_path, metavar="LABEL=PATH"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/report/generalization")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(args.config)
    manifest = load_json(args.dataset_manifest)
    audit = load_json(args.audit)
    dataset_statistics = load_json(args.dataset_statistics)
    trainer_state = load_json(args.trainer_state)
    training_metrics = load_json(args.training_metrics)
    selection = load_json(args.selection_record)

    training_rows = [
        {
            "step": int(item["step"]),
            "epoch": float(item.get("epoch", 0)),
            "loss": item.get("loss", ""),
            "eval_loss": item.get("eval_loss", ""),
            "learning_rate": item.get("learning_rate", ""),
            "grad_norm": item.get("grad_norm", ""),
        }
        for item in trainer_state.get("log_history", [])
        if "loss" in item or "eval_loss" in item
    ]
    write_csv(
        args.output_dir / "training_curve.csv",
        training_rows,
        ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"],
    )
    train_points = [
        (row["step"], float(row["loss"])) for row in training_rows if row["loss"] != ""
    ]
    eval_points = [
        (row["step"], float(row["eval_loss"]))
        for row in training_rows
        if row["eval_loss"] != ""
    ]
    (args.output_dir / "loss_curve.svg").write_text(
        polyline_chart(
            [
                ("Training", COLORS["loss"], train_points),
                ("Validation", COLORS["eval_loss"], eval_points),
            ],
            "Clean generalization training loss",
            "Optimizer step",
            "Completion-only cross-entropy",
        ),
        encoding="utf-8",
    )

    evaluations = []
    for label, path in args.evaluation:
        payload = load_json(path)
        summary = payload["summary"]
        results = payload["results"]
        legal_low, legal_high = wilson(int(summary["legal_moves"]), int(summary["boards"]))
        optimal_low, optimal_high = wilson(
            int(summary["optimal_moves"]), int(summary["boards"])
        )
        score_low, score_high = score_interval(results)
        evaluations.append(
            {
                "label": label,
                "path": str(path),
                "boards": int(summary["boards"]),
                "raw_points": int(summary["raw_points"]),
                "optimal_points": int(summary["optimal_points"]),
                "score_pct": float(summary["score_pct"]),
                "score_ci_low": score_low,
                "score_ci_high": score_high,
                "legal_move_pct": float(summary["legal_move_pct"]),
                "legal_ci_low": legal_low,
                "legal_ci_high": legal_high,
                "optimal_move_pct": float(summary["optimal_move_pct"]),
                "optimal_ci_low": optimal_low,
                "optimal_ci_high": optimal_high,
                "optimal_given_legal_pct": float(summary["optimal_given_legal_pct"]),
                "score_given_legal_pct": float(summary["score_given_legal_pct"]),
                "mean_regret_given_legal": summary["mean_regret_given_legal"],
                "elapsed_seconds": float(summary["elapsed_seconds"]),
                "by_ply": summary["by_ply"],
                "generation": summary["generation"],
            }
        )
    evaluation_fields = [
        key for key in evaluations[0] if key not in {"by_ply", "generation"}
    ] if evaluations else ["label"]
    write_csv(args.output_dir / "evaluation_metrics.csv", evaluations, evaluation_fields)
    (args.output_dir / "evaluation_evolution.svg").write_text(
        grouped_bar_chart(evaluations, float(config["target"]["score_pct"])),
        encoding="utf-8",
    )

    ledger = {
        "config": config,
        "dataset_manifest": manifest,
        "audit": audit,
        "dataset_statistics": dataset_statistics,
        "training_metrics": training_metrics,
        "training_curve": training_rows,
        "checkpoint_selection": selection,
        "evaluations": evaluations,
    }
    (args.output_dir / "experiment_report.json").write_text(
        json.dumps(ledger, indent=2), encoding="utf-8"
    )

    target = config["target"]
    hidden = next((item for item in evaluations if "hidden test" in item["label"].lower()), None)
    official = next((item for item in evaluations if "official" in item["label"].lower()), None)
    lines = [
        "# Leakage-safe Scrabble generalization report",
        "",
        f"Base model: `{config['base_model']}`  ",
        f"Hardware: {config['hardware']['accelerator']} "
        f"({config['hardware']['vram_gb']} GB VRAM)  ",
        f"Uncontaminated closed-model reference supplied by the project owner: "
        f"**approximately {target['score_pct']:.2f}%** ({target['raw_points']}/"
        f"{target['optimal_points']}).",
        "",
        "## Outcome",
        "",
    ]
    if hidden:
        lines.append(
            f"Frozen generated test: **{hidden['raw_points']}/{hidden['optimal_points']} "
            f"({hidden['score_pct']:.2f}%, game-cluster bootstrap 95% CI "
            f"{hidden['score_ci_low']:.2f}–{hidden['score_ci_high']:.2f}%)** across "
            f"{hidden['boards']} positions from games never seen during training or selection."
        )
        lines.append("")
    if official:
        margin = official["score_pct"] - float(target["score_pct"])
        verb = "above" if margin > 0 else "below"
        lines.append(
            f"One-shot official smoke evaluation after checkpoint selection: "
            f"**{official['raw_points']}/{official['optimal_points']} "
            f"({official['score_pct']:.2f}%)**, {abs(margin):.2f} percentage points "
            f"{verb} the owner-supplied reference."
        )
        lines.append("")
    lines.extend(
        [
            "## Leakage controls",
            "",
            f"The machine audit reports `passed: {str(audit['passed']).lower()}`: "
            f"{audit['official_overlap_count']} official overlaps, "
            f"{sum(audit['cross_split_position_duplicates'].values())} cross-split "
            f"position overlaps, {sum(audit['cross_split_game_duplicates'].values())} "
            f"cross-split game overlaps, and {audit['invalid_label_count']} invalid labels.",
            "The official boards were used only once, after the checkpoint was selected on "
            "the 55-position validation subset and evaluated on the frozen generated test.",
            "",
            "## Data",
            "",
            "| Split | Positions | Conversations | Recovery conversations |",
            "|---|---:|---:|---:|",
        ]
    )
    for split in ("train", "validation", "test"):
        item = manifest["splits"][split]
        lines.append(
            f"| {split} | {item['positions']} | {item['records']} | "
            f"{item['recovery_records']} |"
        )
    lines.extend(
        [
            "",
            "| Split | Unique racks | Mean board tiles | Mean optimal score | 95th-percentile optimal score |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in ("train", "validation", "test"):
        item = dataset_statistics[split]
        lines.append(
            f"| {split} | {item['unique_sorted_racks']} | "
            f"{item['mean_board_tiles']:.2f} | "
            f"{item['optimal_score']['mean']:.2f} | "
            f"{item['optimal_score']['p95']} |"
        )
    lines.extend(
        [
            "",
            "## Loss evolution",
            "",
            "![Loss curve](loss_curve.svg)",
            "",
            f"Final train loss: **{float(training_metrics['train_loss']):.6f}**; "
            f"validation loss: **{float(training_metrics['eval_loss']):.6f}**; "
            f"runtime: **{float(training_metrics['train_runtime']) / 3600:.2f} hours**.",
            "",
            "## Checkpoint selection",
            "",
            f"Rule: {'; '.join(selection['rule'])}. Chosen adapter: "
            f"**{selection['chosen']}**. Selection set SHA-256: "
            f"`{selection['selection_sha256']}`.",
            "",
            "## Evaluation evolution",
            "",
            "![Evaluation evolution](evaluation_evolution.svg)",
            "",
            "| Evaluation | Boards | Points | Score | Legal | Exact optimal | Optimal if legal | Mean legal regret |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in evaluations:
        regret = item["mean_regret_given_legal"]
        lines.append(
            f"| {item['label']} | {item['boards']} | {item['raw_points']}/"
            f"{item['optimal_points']} | {item['score_pct']:.2f}% | "
            f"{item['legal_move_pct']:.2f}% | {item['optimal_move_pct']:.2f}% | "
            f"{item['optimal_given_legal_pct']:.2f}% | "
            f"{float(regret):.2f} |" if regret is not None else
            f"| {item['label']} | {item['boards']} | {item['raw_points']}/"
            f"{item['optimal_points']} | {item['score_pct']:.2f}% | "
            f"{item['legal_move_pct']:.2f}% | {item['optimal_move_pct']:.2f}% | "
            f"{item['optimal_given_legal_pct']:.2f}% | n/a |"
        )
    lines.extend(
        [
            "",
            "Machine-readable sources: [complete ledger](experiment_report.json), "
            "[training curve](training_curve.csv), and "
            "[evaluation metrics](evaluation_metrics.csv).",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
