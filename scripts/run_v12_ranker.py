#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scrabble_bench.legal_ranking import evaluate_ranker, train_pairwise_ranker


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    train_inputs = load_jsonl(args.data_dir / "train.inputs.jsonl")
    train_labels = load_jsonl(args.data_dir / "train.labels.jsonl")
    validation_inputs = load_jsonl(args.data_dir / "validation.inputs.jsonl")
    validation_labels = load_jsonl(args.data_dir / "validation.labels.jsonl")
    model = train_pairwise_ranker(
        train_inputs,
        train_labels,
        seed=args.seed,
        epochs=args.epochs,
        negatives_per_board=args.negatives_per_board,
    )
    result = evaluate_ranker(validation_inputs, validation_labels, model, seed=args.seed + 1)
    write(args.output_dir / "model.json", model)
    write(args.output_dir / "validation.json", result)
    print(json.dumps({"model": model["training"], "validation": {key: value for key, value in result.items() if key != "details"}}, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    model = json.loads(args.model.read_text(encoding="utf-8"))
    inputs = load_jsonl(args.inputs)
    labels = load_jsonl(args.labels)
    result = evaluate_ranker(inputs, labels, model, seed=args.seed)
    write(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train or evaluate the CPU V12 score-blind ranker.")
    commands = result.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train and evaluate on validation only.")
    train_parser.add_argument("--data-dir", type=Path, default=ROOT / "data/v12_legal_ranking")
    train_parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/v12_legal_ranking")
    train_parser.add_argument("--seed", type=int, default=12012)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--negatives-per-board", type=int, default=64)
    train_parser.set_defaults(function=train)
    evaluate_parser = commands.add_parser("evaluate", help="Explicitly evaluate a frozen model on named inputs.")
    evaluate_parser.add_argument("--model", type=Path, required=True)
    evaluate_parser.add_argument("--inputs", type=Path, required=True)
    evaluate_parser.add_argument("--labels", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--seed", type=int, default=12013)
    evaluate_parser.set_defaults(function=evaluate)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)

