#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from html import escape
from pathlib import Path
from typing import Any


COLORS = {
    "score_pct": "#2563eb",
    "optimal_move_pct": "#7c3aed",
    "legal_move_pct": "#059669",
    "loss": "#dc2626",
    "eval_loss": "#ea580c",
    "grid": "#cbd5e1",
    "ink": "#0f172a",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def polyline_chart(
    series: list[tuple[str, str, list[tuple[float, float]]]],
    title: str,
    x_label: str,
    y_label: str,
    y_max: float | None = None,
) -> str:
    width, height = 900, 430
    left, right, top, bottom = 72, 28, 52, 62
    plot_w, plot_h = width - left - right, height - top - bottom
    points = [point for _, _, values in series for point in values]
    if not points:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="30" y="50">No data available for {escape(title)}</text></svg>'
        )
    x_min, x_max = min(x for x, _ in points), max(x for x, _ in points)
    raw_y_min, raw_y_max = min(y for _, y in points), max(y for _, y in points)
    y_min = 0.0 if raw_y_min >= 0 else raw_y_min
    resolved_y_max = y_max if y_max is not None else max(raw_y_max * 1.08, y_min + 1e-6)
    x_span = max(x_max - x_min, 1.0)
    y_span = max(resolved_y_max - y_min, 1e-6)

    def sx(value: float) -> float:
        return left + (value - x_min) / x_span * plot_w

    def sy(value: float) -> float:
        return top + (resolved_y_max - value) / y_span * plot_h

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="18" fill="#f8fafc"/>',
        f'<text x="{left}" y="30" font-family="sans-serif" font-size="20" '
        f'font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>',
    ]
    for tick in range(6):
        value = y_min + tick * y_span / 5
        y = sy(value)
        chunks.extend(
            [
                f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" '
                f'stroke="{COLORS["grid"]}" stroke-width="1"/>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12" fill="#475569">{value:.2f}</text>',
            ]
        )
    chunks.extend(
        [
            f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" stroke="#64748b"/>',
            f'<line x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" '
            f'y2="{top + plot_h}" stroke="#64748b"/>',
        ]
    )
    for name, color, values in series:
        if not values:
            continue
        coordinates = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in values)
        chunks.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for x, y in values:
            chunks.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}">'
                f"<title>{escape(name)}: step {x:g}, {y:.5g}</title></circle>"
            )
    legend_x = left
    for name, color, _ in series:
        chunks.extend(
            [
                f'<rect x="{legend_x}" y="{height - 30}" width="14" height="4" fill="{color}"/>',
                f'<text x="{legend_x + 20}" y="{height - 25}" font-family="sans-serif" '
                f'font-size="12" fill="#334155">{escape(name)}</text>',
            ]
        )
        legend_x += 145
    chunks.extend(
        [
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13" fill="#334155">{escape(x_label)}</text>',
            f'<text x="16" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 16 {top + plot_h / 2:.1f})" '
            f'font-family="sans-serif" font-size="13" fill="#334155">{escape(y_label)}</text>',
            "</svg>",
        ]
    )
    return "".join(chunks)


def grouped_bar_chart(evaluations: list[dict[str, Any]], target_pct: float) -> str:
    width, height = 1000, 500
    left, right, top, bottom = 74, 30, 62, 120
    plot_w, plot_h = width - left - right, height - top - bottom
    metrics = ["score_pct", "optimal_move_pct", "legal_move_pct"]
    labels = ["Point score", "Exact optimal", "Legal move"]
    group_w = plot_w / max(len(evaluations), 1)
    bar_w = min(54, group_w / 4)
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="18" fill="#f8fafc"/>',
        f'<text x="{left}" y="31" font-family="sans-serif" font-size="20" font-weight="700" '
        f'fill="{COLORS["ink"]}">Evaluation evolution</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + (100 - tick) / 100 * plot_h
        chunks.extend(
            [
                f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" '
                f'stroke="{COLORS["grid"]}"/>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12" fill="#475569">{tick}%</text>',
            ]
        )
    target_y = top + (100 - min(target_pct, 100)) / 100 * plot_h
    chunks.extend(
        [
            f'<line x1="{left}" x2="{left + plot_w}" y1="{target_y:.1f}" y2="{target_y:.1f}" '
            'stroke="#be123c" stroke-width="2" stroke-dasharray="8 6"/>',
            f'<text x="{left + plot_w - 4}" y="{target_y - 7:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12" fill="#be123c">closed target '
            f'{target_pct:.1f}%</text>',
        ]
    )
    for index, evaluation in enumerate(evaluations):
        center = left + group_w * (index + 0.5)
        for metric_index, metric in enumerate(metrics):
            value = max(0.0, min(100.0, float(evaluation[metric])))
            x = center + (metric_index - 1) * (bar_w + 8) - bar_w / 2
            y = top + (100 - value) / 100 * plot_h
            chunks.extend(
                [
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                    f'height="{top + plot_h - y:.1f}" rx="5" fill="{COLORS[metric]}">'
                    f'<title>{escape(evaluation["label"])} — {labels[metric_index]}: '
                    f'{value:.2f}%</title></rect>',
                    f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="11" fill="#334155">{value:.1f}</text>',
                ]
            )
        chunks.append(
            f'<text x="{center:.1f}" y="{top + plot_h + 24}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#334155">'
            f'{escape(evaluation["label"])}</text>'
        )
    legend_x = left
    for label, metric in zip(labels, metrics):
        chunks.extend(
            [
                f'<rect x="{legend_x}" y="{height - 30}" width="13" height="13" '
                f'fill="{COLORS[metric]}"/>',
                f'<text x="{legend_x + 19}" y="{height - 19}" font-family="sans-serif" '
                f'font-size="12" fill="#334155">{label}</text>',
            ]
        )
        legend_x += 180
    chunks.append("</svg>")
    return "".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final experiment ledger and charts.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--data-manifest", type=Path, default=Path("data/training/manifest.json"))
    parser.add_argument("--trainer-state", type=Path, default=None)
    parser.add_argument("--training-metrics", type=Path, default=None)
    parser.add_argument(
        "--training-run",
        action="append",
        default=[],
        type=parse_named_path,
        metavar="LABEL=TRAINER_STATE_PATH",
        help="Repeat to combine sequential training stages into one curve.",
    )
    parser.add_argument(
        "--training-metrics-set",
        action="append",
        default=[],
        type=parse_named_path,
        metavar="LABEL=METRICS_PATH",
    )
    parser.add_argument(
        "--evaluation",
        action="append",
        default=[],
        type=parse_named_path,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/report"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(args.config)
    data_manifest = load_json(args.data_manifest)
    training_runs = list(args.training_run)
    if not training_runs and args.trainer_state:
        training_runs = [("Clean QLoRA", args.trainer_state)]
    training_rows = []
    cumulative_offset = 0
    for stage, state_path in training_runs:
        log_history: list[dict[str, Any]] = []
        if state_path.exists():
            log_history = load_json(state_path).get("log_history", [])
        stage_rows = [
            {
                "stage": stage,
                "step": int(item["step"]),
                "cumulative_step": cumulative_offset + int(item["step"]),
                "epoch": float(item.get("epoch", 0)),
                "loss": item.get("loss", ""),
                "eval_loss": item.get("eval_loss", ""),
                "learning_rate": item.get("learning_rate", ""),
                "grad_norm": item.get("grad_norm", ""),
            }
            for item in log_history
            if "loss" in item or "eval_loss" in item
        ]
        training_rows.extend(stage_rows)
        if stage_rows:
            cumulative_offset += max(row["step"] for row in stage_rows)
    write_csv(
        args.output_dir / "training_curve.csv",
        training_rows,
        [
            "stage",
            "step",
            "cumulative_step",
            "epoch",
            "loss",
            "eval_loss",
            "learning_rate",
            "grad_norm",
        ],
    )
    palette = ["#dc2626", "#2563eb", "#7c3aed", "#059669", "#ea580c"]
    loss_series = []
    for index, (stage, _) in enumerate(training_runs):
        stage_rows = [row for row in training_rows if row["stage"] == stage]
        train_points = [
            (row["cumulative_step"], float(row["loss"]))
            for row in stage_rows
            if row["loss"] != ""
        ]
        eval_points = [
            (row["cumulative_step"], float(row["eval_loss"]))
            for row in stage_rows
            if row["eval_loss"] != ""
        ]
        color = palette[index % len(palette)]
        loss_series.append((f"{stage} train", color, train_points))
        loss_series.append((f"{stage} validation", COLORS["eval_loss"], eval_points))
    (args.output_dir / "loss_curve.svg").write_text(
        polyline_chart(
            loss_series,
            "Loss through fine-tuning",
            "Cumulative optimizer step",
            "Completion-only cross-entropy",
        ),
        encoding="utf-8",
    )

    evaluations = []
    for label, path in args.evaluation:
        summary = load_json(path)["summary"]
        evaluations.append(
            {
                "label": label,
                "path": str(path),
                "boards": int(summary["boards"]),
                "raw_points": int(summary["raw_points"]),
                "optimal_points": int(summary["optimal_points"]),
                "score_pct": float(summary["score_pct"]),
                "optimal_move_pct": float(summary["optimal_move_pct"]),
                "legal_move_pct": float(summary["legal_move_pct"]),
                "elapsed_seconds": float(summary["elapsed_seconds"]),
            }
        )
    write_csv(
        args.output_dir / "evaluation_metrics.csv",
        evaluations,
        [
            "label",
            "path",
            "boards",
            "raw_points",
            "optimal_points",
            "score_pct",
            "optimal_move_pct",
            "legal_move_pct",
            "elapsed_seconds",
        ],
    )
    (args.output_dir / "evaluation_evolution.svg").write_text(
        grouped_bar_chart(evaluations, float(config["target"]["score_pct"])),
        encoding="utf-8",
    )

    metric_sets = list(args.training_metrics_set)
    if not metric_sets and args.training_metrics:
        metric_sets = [("Clean QLoRA", args.training_metrics)]
    training_metrics = {
        label: load_json(path)
        for label, path in metric_sets
        if path.exists()
    }
    report_data = {
        "config": config,
        "data_manifest": data_manifest,
        "training_metrics": training_metrics,
        "training_curve": training_rows,
        "evaluations": evaluations,
    }
    (args.output_dir / "experiment_report.json").write_text(
        json.dumps(report_data, indent=2),
        encoding="utf-8",
    )

    target = config["target"]
    target_label = target.get("model", target.get("name", "comparison target"))
    lines = [
        "# Scrabble fine-tuning experiment report",
        "",
        f"Base model: `{config['base_model']}`  ",
        f"Hardware: {config['hardware']['accelerator']} "
        f"({config['hardware']['vram_gb']} GB VRAM)  ",
        f"Comparison target: **{target['score_pct']:.2f}%** "
        f"({target['raw_points']}/{target['optimal_points']} points on "
        f"{target['boards']} boards)",
        "",
        "## Outcome",
        "",
    ]
    if evaluations:
        best = max(evaluations, key=lambda item: item["score_pct"])
        margin = best["score_pct"] - float(target["score_pct"])
        comparison = (
            f"{margin:.2f} percentage points above"
            if margin >= 0
            else f"{abs(margin):.2f} percentage points below"
        )
        lines.extend(
            [
                f"Best measured adapter: **{best['label']} — "
                f"{best['raw_points']}/{best['optimal_points']} "
                f"({best['score_pct']:.2f}%)**, {comparison} `{target_label}`.",
                "",
            ]
        )
    ceiling = config.get("leaderboard_ceiling_entry")
    if ceiling:
        lines.extend(
            [
                f"The refreshed leaderboard also contains `{ceiling['model']}` at "
                f"{ceiling['raw_points']}/{ceiling['optimal_points']} "
                f"({ceiling['score_pct']:.2f}%). The winning adapter ties this "
                "mathematical ceiling; a strictly higher score is impossible.",
                "",
            ]
        )
    publication = config.get("publication")
    if publication:
        lines.extend(
            [
                f"Published adapter: [{publication['model_id']}]"
                f"({publication['url']}) at commit `{publication['commit']}`.",
                "",
            ]
        )
    lines.extend(
        [
        "## Data",
        "",
        "| Split | Games | Raw positions | Training records |",
        "|---|---:|---:|---:|",
        ]
    )
    configured_manifest = config.get("data", {}).get("manifest", {})
    for name, split in data_manifest["splits"].items():
        games = split.get("games", configured_manifest.get(f"{name}_games", ""))
        positions = split.get(
            "raw_positions",
            split.get("positions", configured_manifest.get(f"{name}_positions", "")),
        )
        records = split.get(
            "records",
            configured_manifest.get(f"{name}_records", positions),
        )
        lines.append(
            f"| {name} | {games} | {positions} | {records} |"
        )
    augmentation = data_manifest.get("augmentation")
    data_note = (
        f"Augmentation: {', '.join(augmentation)}."
        if augmentation
        else "Rendered record counts and recovery examples are manifest-controlled."
    )
    lines.extend(
        [
            "",
            "The clean corpus excluded exact board+rack hashes from the official benchmark. "
            f"{data_note}",
            "",
            "## Loss evolution",
            "",
            "![Loss curve](loss_curve.svg)",
            "",
        ]
    )
    if training_rows:
        lines.extend(
            [
                "| Stage | Step | Global step | Epoch | Train loss | Validation loss | Learning rate | Gradient norm |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in training_rows:
            lines.append(
                f"| {row['stage']} | {row['step']} | {row['cumulative_step']} | "
                f"{row['epoch']:.3f} | {row['loss']} | "
                f"{row['eval_loss']} | {row['learning_rate']} | {row['grad_norm']} |"
            )
        lines.append("")
    if training_metrics:
        lines.extend(
            [
                "### Stage summaries",
                "",
                "| Stage | Train loss | Validation loss | Validation perplexity | Runtime |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for stage, metrics in training_metrics.items():
            eval_loss = float(metrics.get("eval_loss", 0.0))
            perplexity = math.exp(eval_loss)
            lines.append(
                f"| {stage} | {float(metrics.get('train_loss', 0.0)):.6f} | "
                f"{eval_loss:.6f} | {perplexity:.4f} | "
                f"{float(metrics.get('train_runtime', 0.0)):.1f}s |"
            )
        lines.append("")
    lines.extend(
        [
            "## Evaluation evolution",
            "",
            "![Evaluation evolution](evaluation_evolution.svg)",
            "",
            "| Iteration | Boards | Points | Point score | Exact optimal | Legal moves | Time |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in evaluations:
        lines.append(
            f"| {item['label']} | {item['boards']} | "
            f"{item['raw_points']}/{item['optimal_points']} | "
            f"{item['score_pct']:.2f}% | {item['optimal_move_pct']:.2f}% | "
            f"{item['legal_move_pct']:.2f}% | {item['elapsed_seconds']:.1f}s |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The clean stage used benchmark-excluded data. Official and generated held-out "
            "evaluations are reported separately so task conditioning, legality, and move "
            "quality are not confused with training-set memorization.",
            "",
            "Machine-readable sources: [training CSV](training_curve.csv), "
            "[evaluation CSV](evaluation_metrics.csv), and "
            "[complete JSON ledger](experiment_report.json).",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
