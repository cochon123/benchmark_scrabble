#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from html import escape
from pathlib import Path
from typing import Any

from scrabble_bench.evaluation import summarize_evaluation


COLORS = {
    "score_pct": "#2563eb",
    "legal_move_pct": "#059669",
    "optimal_move_pct": "#7c3aed",
}


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def render_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 1000, 520
    left, right, top, bottom = 72, 32, 58, 105
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = max(
        [float(row[key]) for row in rows for key in COLORS] + [0.0]
    )
    y_max = min(100.0, max(10.0, math.ceil(maximum / 10.0) * 10.0))

    def x(index: int) -> float:
        return left + index * plot_width / max(len(rows) - 1, 1)

    def y(value: float) -> float:
        return top + (y_max - value) / y_max * plot_height

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="18" fill="#f8fafc"/>',
        f'<text x="{left}" y="32" font-family="sans-serif" font-size="21" font-weight="700" fill="#0f172a">V3 five-board evolution</text>',
    ]
    for tick in range(6):
        value = tick * y_max / 5
        tick_y = y(value)
        chunks.extend(
            [
                f'<line x1="{left}" x2="{left + plot_width}" y1="{tick_y:.1f}" y2="{tick_y:.1f}" stroke="#cbd5e1"/>',
                f'<text x="{left - 10}" y="{tick_y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#475569">{value:.1f}%</text>',
            ]
        )
    for metric, color in COLORS.items():
        points = " ".join(
            f"{x(index):.1f},{y(float(row[metric])):.1f}"
            for index, row in enumerate(rows)
        )
        chunks.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for index, row in enumerate(rows):
            value = float(row[metric])
            chunks.append(
                f'<circle cx="{x(index):.1f}" cy="{y(value):.1f}" r="4" fill="{color}"><title>{escape(str(row["label"]))} — {metric}: {value:.3f}%</title></circle>'
            )
    for index, row in enumerate(rows):
        chunks.append(
            f'<text x="{x(index):.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#334155">{escape(str(row["label"]))}</text>'
        )
    legend = [
        ("Point score", COLORS["score_pct"]),
        ("Legal moves", COLORS["legal_move_pct"]),
        ("Exact optimal", COLORS["optimal_move_pct"]),
    ]
    legend_x = left
    for name, color in legend:
        chunks.extend(
            [
                f'<rect x="{legend_x}" y="{height - 31}" width="14" height="5" fill="{color}"/>',
                f'<text x="{legend_x + 21}" y="{height - 25}" font-family="sans-serif" font-size="12" fill="#334155">{escape(name)}</text>',
            ]
        )
        legend_x += 165
    chunks.append("</svg>")
    return "".join(chunks)


def training_loss_rows(path: Path) -> list[dict[str, Any]]:
    state = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "step": int(item["step"]),
            "loss": float(item["loss"]),
            "learning_rate": float(item.get("learning_rate", 0.0)),
            "grad_norm": float(item.get("grad_norm", 0.0)),
            "epoch": float(item.get("epoch", 0.0)),
        }
        for item in state.get("log_history", [])
        if "step" in item and "loss" in item
    ]


def render_loss_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 1000, 440
    left, right, top, bottom = 72, 32, 58, 72
    plot_width, plot_height = width - left - right, height - top - bottom
    if not rows:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            '<text x="30" y="50" font-family="sans-serif">No training loss data.</text></svg>'
        )
    x_min, x_max = rows[0]["step"], rows[-1]["step"]
    y_max = max(float(row["loss"]) for row in rows) * 1.05

    def x(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1) * plot_width

    def y(value: float) -> float:
        return top + (y_max - value) / max(y_max, 1e-9) * plot_height

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="18" fill="#f8fafc"/>',
        f'<text x="{left}" y="32" font-family="sans-serif" font-size="21" font-weight="700" fill="#0f172a">V3 weighted completion loss</text>',
    ]
    for tick in range(6):
        value = tick * y_max / 5
        tick_y = y(value)
        chunks.extend(
            [
                f'<line x1="{left}" x2="{left + plot_width}" y1="{tick_y:.1f}" y2="{tick_y:.1f}" stroke="#cbd5e1"/>',
                f'<text x="{left - 10}" y="{tick_y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#475569">{value:.3f}</text>',
            ]
        )
    points = " ".join(
        f'{x(float(row["step"])):.1f},{y(float(row["loss"])):.1f}' for row in rows
    )
    chunks.append(
        f'<polyline points="{points}" fill="none" stroke="#dc2626" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    for row in rows:
        chunks.append(
            f'<circle cx="{x(float(row["step"])):.1f}" cy="{y(float(row["loss"])):.1f}" r="3" fill="#dc2626"><title>step {row["step"]}: loss {row["loss"]:.6f}</title></circle>'
        )
    chunks.extend(
        [
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#334155">Optimizer step</text>',
            f'<text x="17" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 17 {top + plot_height / 2:.1f})" font-family="sans-serif" font-size="13" fill="#334155">Weighted completion loss</text>',
            "</svg>",
        ]
    )
    return "".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen five-board v3 evolution chart.")
    parser.add_argument("--evaluation", action="append", required=True, type=named_path)
    parser.add_argument("--trainer-state", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for label, path in args.evaluation:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        legality = summarize_evaluation(payload)["final"]
        rows.append(
            {
                "label": label,
                "path": str(path),
                "boards": int(summary["boards"]),
                "raw_points": int(summary["raw_points"]),
                "optimal_points": int(summary["optimal_points"]),
                "score_pct": float(summary["score_pct"]),
                "legal_moves": int(summary["legal_moves"]),
                "legal_move_pct": float(summary["legal_move_pct"]),
                "optimal_moves": int(summary["optimal_moves"]),
                "optimal_move_pct": float(summary["optimal_move_pct"]),
                "recovered_after_rejection": int(legality["recovered_after_rejection"]),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evolution_metrics.json").write_text(
        json.dumps({"evaluations": rows}, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "evolution_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "evolution_chart.svg").write_text(render_svg(rows), encoding="utf-8")
    report: dict[str, Any] = {"evaluations": rows}
    if args.trainer_state:
        losses = training_loss_rows(args.trainer_state)
        if not losses:
            raise RuntimeError(f"No loss rows in trainer state: {args.trainer_state}")
        with (args.output_dir / "training_loss.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(losses[0]))
            writer.writeheader()
            writer.writerows(losses)
        (args.output_dir / "training_loss.json").write_text(
            json.dumps({"training_loss": losses}, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "training_loss_chart.svg").write_text(
            render_loss_svg(losses), encoding="utf-8"
        )
        report["training_loss"] = losses
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
