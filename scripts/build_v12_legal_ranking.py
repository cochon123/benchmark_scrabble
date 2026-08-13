#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.legal_ranking import (
    baseline_metrics,
    convert_frozen_board,
    enumerate_board,
    game_ids,
    select_game_separated,
    sha256,
)
from scrabble_bench.lexicon import Lexicon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN_SHA256 = "413c6ed82a790697052f16f565a822a907302c5acfe931f30ac2a18f864e54cf"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_split(
    rows: list[dict[str, Any]],
    lexicon: Lexicon,
    *,
    split: str,
    limit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = select_game_separated(rows, limit, seed=seed) if limit else rows
    inputs: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    started = time.time()
    for index, position in enumerate(selected, 1):
        public, hidden = enumerate_board(position, lexicon)
        inputs.append(public)
        labels.append(hidden)
        if index == 1 or index % 10 == 0 or index == len(selected):
            print(f"{split}: enumerated {index}/{len(selected)}", flush=True)
    print(f"{split}: {time.time() - started:.1f}s", flush=True)
    return inputs, labels


def build(args: argparse.Namespace) -> None:
    train_rows = read_json(args.train_positions)
    validation_rows = read_json(args.validation_positions)
    frozen = read_json(args.frozen_test_candidates)
    observed_sha = sha256(args.frozen_test_candidates)
    if observed_sha != args.frozen_test_sha256:
        raise RuntimeError(f"Frozen test hash changed: {observed_sha} != {args.frozen_test_sha256}")
    test_rows = frozen["boards"]
    train_games = game_ids(train_rows)
    validation_games = game_ids(validation_rows)
    test_games = game_ids(test_rows)
    overlaps = {
        "train_validation": sorted(train_games & validation_games),
        "train_test": sorted(train_games & test_games),
        "validation_test": sorted(validation_games & test_games),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Whole-game leakage detected: {overlaps}")

    lexicon = Lexicon.from_path(resolve_lexicon_path())
    train_inputs, train_labels = build_split(
        train_rows, lexicon, split="train", limit=args.train_boards, seed=args.seed
    )
    validation_inputs, validation_labels = build_split(
        validation_rows, lexicon, split="validation", limit=args.validation_boards, seed=args.seed + 1
    )
    test_inputs: list[dict[str, Any]] = []
    test_labels: list[dict[str, Any]] = []
    selected_test = test_rows[: args.test_boards] if args.test_boards else test_rows
    for index, row in enumerate(selected_test, 1):
        public, hidden = convert_frozen_board(row, lexicon)
        test_inputs.append(public)
        test_labels.append(hidden)
        if index == 1 or index % 10 == 0 or index == len(selected_test):
            print(f"test: converted {index}/{len(selected_test)}", flush=True)

    for split, inputs, labels in (
        ("train", train_inputs, train_labels),
        ("validation", validation_inputs, validation_labels),
        ("test", test_inputs, test_labels),
    ):
        write_jsonl(args.output_dir / f"{split}.inputs.jsonl", inputs)
        write_jsonl(args.output_dir / f"{split}.labels.jsonl", labels)

    baselines = {
        split: baseline_metrics(inputs, labels)
        for split, inputs, labels in (
            ("train", train_inputs, train_labels),
            ("validation", validation_inputs, validation_labels),
            ("test", test_inputs, test_labels),
        )
    }
    write_json(args.output_dir / "baselines.json", baselines)
    manifest = {
        "version": "v12-legal-ranking-pilot-1",
        "contract": "docs/V12_LEGAL_RANKING.md",
        "seed": args.seed,
        "score_blind_inputs": True,
        "labels_are_evaluator_only": True,
        "whole_game_overlap": {name: len(value) for name, value in overlaps.items()},
        "sources": {
            "train": {"path": str(args.train_positions), "sha256": sha256(args.train_positions)},
            "validation": {"path": str(args.validation_positions), "sha256": sha256(args.validation_positions)},
            "test": {"path": str(args.frozen_test_candidates), "sha256": observed_sha, "immutable": True},
        },
        "boards": {"train": len(train_inputs), "validation": len(validation_inputs), "test": len(test_inputs)},
        "baselines": baselines,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def baselines(args: argparse.Namespace) -> None:
    inputs = load_jsonl(args.inputs)
    labels = load_jsonl(args.labels)
    print(json.dumps(baseline_metrics(inputs, labels), indent=2))


def prepare_positions(args: argparse.Namespace) -> None:
    observed_sha = sha256(args.positions)
    if args.expected_sha256 and observed_sha != args.expected_sha256:
        raise RuntimeError(f"Position file hash changed: {observed_sha} != {args.expected_sha256}")
    rows = read_json(args.positions)
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    inputs, labels = build_split(rows, lexicon, split=args.split, limit=args.boards, seed=args.seed)
    write_jsonl(args.output_dir / f"{args.split}.inputs.jsonl", inputs)
    write_jsonl(args.output_dir / f"{args.split}.labels.jsonl", labels)
    payload = {
        "split": args.split,
        "source": {"path": str(args.positions), "sha256": observed_sha},
        "boards": len(inputs),
        "unique_games": len(game_ids(rows)),
        "score_blind_inputs": True,
        "labels_are_evaluator_only": True,
        "baselines": baseline_metrics(inputs, labels),
    }
    write_json(args.output_dir / f"{args.split}.manifest.json", payload)
    print(json.dumps(payload, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build score-blind V12 full-legal-space ranking data.")
    subparsers = result.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--train-positions", type=Path, default=ROOT / "data/general_v6_positions/train_positions.json")
    build_parser.add_argument("--validation-positions", type=Path, default=ROOT / "data/general_v8_fresh/validation_positions.json")
    build_parser.add_argument("--frozen-test-candidates", type=Path, default=ROOT / "artifacts/v8/ranking/candidates-300.json")
    build_parser.add_argument("--frozen-test-sha256", default=DEFAULT_FROZEN_SHA256)
    build_parser.add_argument("--output-dir", type=Path, default=ROOT / "data/v12_legal_ranking")
    build_parser.add_argument("--train-boards", type=int, default=240)
    build_parser.add_argument("--validation-boards", type=int, default=0, help="0 uses every validation position")
    build_parser.add_argument("--test-boards", type=int, default=0, help="0 uses all 300 immutable test boards")
    build_parser.add_argument("--seed", type=int, default=12012)
    build_parser.set_defaults(function=build)
    baseline_parser = subparsers.add_parser("baselines")
    baseline_parser.add_argument("--inputs", type=Path, required=True)
    baseline_parser.add_argument("--labels", type=Path, required=True)
    baseline_parser.set_defaults(function=baselines)
    prepare_parser = subparsers.add_parser("prepare-positions", help="Prepare a named frozen confirmation set.")
    prepare_parser.add_argument("--positions", type=Path, required=True)
    prepare_parser.add_argument("--expected-sha256")
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--split", default="confirmation")
    prepare_parser.add_argument("--boards", type=int, default=0, help="0 uses every position")
    prepare_parser.add_argument("--seed", type=int, default=12014)
    prepare_parser.set_defaults(function=prepare_positions)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
