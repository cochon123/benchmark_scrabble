#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_line(name: str, payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    regret = summary["mean_regret_given_legal"]
    return (
        f"| {name} | {summary['boards']} | {summary['raw_points']}/"
        f"{summary['optimal_points']} ({summary['score_pct']:.2f}%) | "
        f"{summary['legal_move_pct']:.2f}% | {summary['optimal_move_pct']:.2f}% | "
        f"{summary['optimal_given_legal_pct']:.2f}% | "
        f"{float(regret):.2f} |" if regret is not None else
        f"| {name} | {summary['boards']} | {summary['raw_points']}/"
        f"{summary['optimal_points']} ({summary['score_pct']:.2f}%) | "
        f"{summary['legal_move_pct']:.2f}% | {summary['optimal_move_pct']:.2f}% | "
        f"{summary['optimal_given_legal_pct']:.2f}% | n/a |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the selected adapter model card.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--base-selection", type=Path, required=True)
    parser.add_argument("--selected-selection", type=Path, required=True)
    parser.add_argument("--hidden-test", type=Path, required=True)
    parser.add_argument("--official-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load(args.config)
    manifest = load(args.manifest)
    audit = load(args.audit)
    selection = load(args.selection)
    evaluations = [
        ("Base — validation selection", load(args.base_selection)),
        ("Selected adapter — validation selection", load(args.selected_selection)),
        ("Selected adapter — frozen generated test", load(args.hidden_test)),
        ("Selected adapter — official smoke (one shot)", load(args.official_smoke)),
    ]
    target = config["target"]
    official = evaluations[-1][1]["summary"]
    hidden = evaluations[-2][1]["summary"]
    lines = [
        "---",
        f"base_model: {config['base_model']}",
        "library_name: peft",
        "license: apache-2.0",
        "language:",
        "- en",
        "tags:",
        "- scrabble",
        "- qlora",
        "- reasoning",
        "- game-playing",
        "---",
        "",
        "# Qwen3 4B Scrabble General",
        "",
        "This is a QLoRA adapter trained to choose high-scoring legal Scrabble moves from "
        "a board and rack. Unlike the earlier smoke-specialized memorization control, this "
        "adapter was trained and selected without any official benchmark position.",
        "",
        "## Result",
        "",
        f"The selected checkpoint is `{selection['chosen']}`. On 440 positions from 40 "
        f"entirely held-out generated games it scored **{hidden['raw_points']}/"
        f"{hidden['optimal_points']} ({hidden['score_pct']:.2f}%)**, with "
        f"{hidden['legal_move_pct']:.2f}% legal moves. The official five-board smoke run, "
        f"executed once after selection, scored **{official['raw_points']}/"
        f"{official['optimal_points']} ({official['score_pct']:.2f}%)**. The project "
        f"owner's uncontaminated closed-model reference is approximately "
        f"{target['score_pct']:.2f}% ({target['raw_points']}/{target['optimal_points']}).",
        "",
        "| Evaluation | Boards | Point score | Legal | Exact optimal | Optimal if legal | Mean legal regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[metric_line(name, payload) for name, payload in evaluations],
        "",
        "All evaluations use three validator-feedback attempts, sampling temperature 0.6, "
        "top-p 0.95, top-k 20, and seed 3407.",
        "",
        "## Leakage controls",
        "",
        f"The source audit passed: {audit['official_overlap_count']} official board+rack "
        f"overlaps, {sum(audit['cross_split_position_duplicates'].values())} cross-split "
        f"position overlaps, {sum(audit['cross_split_game_duplicates'].values())} "
        f"cross-split game overlaps, and {audit['invalid_label_count']} invalid labels. "
        "Games—not individual positions—were split before record construction. The "
        "55-position checkpoint-selection set comes only from validation games. The frozen "
        "generated test and official benchmark were not used to choose the checkpoint.",
        "",
        "## Training",
        "",
        f"Training used {manifest['splits']['train']['records']} conversations derived from "
        f"{manifest['splits']['train']['positions']} positions. Of these, "
        f"{manifest['splits']['train']['recovery_records']} teach recovery from an invalid "
        "move using the benchmark validator's exact feedback. Each target contains a short "
        "solver-derived reasoning trace and the final raw `play_move` JSON. Loss is applied "
        "only to assistant tokens.",
        "",
        "QLoRA configuration: 4-bit NF4 with double quantization, rank 16, alpha 32, "
        "all-linear targets, learning rate 1e-4, effective batch size 4, maximum length "
        "2,816, activation checkpointing, and seed 3407.",
        "",
        "## Use",
        "",
        "```python",
        "from peft import PeftModel",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f"base = \"{config['base_model']}\"",
        "adapter = \"Cochon123/Qwen3-4B-Scrabble-General\"",
        "tokenizer = AutoTokenizer.from_pretrained(adapter)",
        "model = AutoModelForCausalLM.from_pretrained(base, device_map=\"auto\")",
        "model = PeftModel.from_pretrained(model, adapter)",
        "```",
        "",
        "Use the benchmark's `prompt_for_position` schema and parse the final JSON object. "
        "This adapter is for highest immediate raw score, not full-game equity or tournament "
        "strategy.",
        "",
        "## Limitations",
        "",
        "The model can still make illegal moves or miss the optimum. It is tied to the "
        "project's English lexicon, coordinate convention, prompt format, and immediate-score "
        "objective. Sampling is reproducible only with the recorded software stack and seed. "
        "The official smoke set has only five boards; the 440-position game-held-out test is "
        "the stronger generalization measurement.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
