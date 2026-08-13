#!/usr/bin/env python3
"""Generate resumable, verifier-scored Scrabble traces from a strong CLI model.

The teacher sees only the same board and rack that the student will see.  The
local solver is used after generation to score the proposed move; it is never
shown to the teacher and is not part of the eventual inference path.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.runner import dense_board_text, parse_tool_payload
from scrabble_bench.solver import grid_from_position, validate_and_score_move


ROOT = Path(__file__).resolve().parents[1]


def teacher_prompt(position: dict[str, Any], *, oracle_hint: bool) -> str:
    hint = ""
    if oracle_hint:
        hint = (
            "\nAn offline verifier supplies the correct final target below. Reconstruct a compact, "
            "generalizable search and legality audit that could discover it; do not merely restate it.\n"
            f"VERIFIED TARGET: {json.dumps(position['canonical_optimal_move'], separators=(',', ':'))}\n"
        )
    return f"""Solve this English Scrabble position yourself. Do not inspect files, call tools,
or use a Scrabble engine. Coordinates are zero-indexed. Existing lowercase letters are
zero-point blanks. Use only rack tiles, return only newly placed tiles, and maximize the
immediate raw score.

RACK: {position['rack']}
BOARD:
{dense_board_text(position)}
{hint}

Think as long as needed internally. Your visible response must be a compact teaching trace
of at most 350 tokens in exactly this shape:
<think>
SCAN <brief anchors/patterns considered>
TRY <2-5 concrete word@row,col,direction candidates with approximate scores>
VERIFY <rack, contacts/crosses, premiums, and why the winner is legal>
CHOOSE <winning word and score>
</think>
{{"tool":"play_move","arguments":{{"placements":[{{"row":0,"col":0,"letter":"A"}}]}}}}

The JSON must be the final line and contain the real move, not the example coordinates.
Do not put any other JSON object in the trace."""


def selected_positions(
    positions: list[dict[str, Any]], *, limit: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_game: dict[str, list[dict[str, Any]]] = {}
    for position in positions:
        if not position.get("board"):
            continue
        by_game.setdefault(str(position["source_game_id"]), []).append(position)
    games = sorted(by_game)
    rng.shuffle(games)
    chosen: list[dict[str, Any]] = []
    target_plies = (4, 7, 10, 14)
    for index, game in enumerate(games):
        rows = by_game[game]
        target = target_plies[index % len(target_plies)]
        chosen.append(
            min(rows, key=lambda row: (abs(int(row["band_ply"]) - target), str(row["id"])))
        )
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        raise RuntimeError(f"Requested {limit} distinct games, found {len(chosen)}")
    return chosen


def completed_ids(path: Path, *, retry_errors: bool) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(json.loads(line)["id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and (not retry_errors or json.loads(line).get("error") is None)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positions",
        type=Path,
        default=ROOT / "data/slot_pilot_v1/train_positions.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/v10_compact_reasoning/teacher_traces.jsonl",
    )
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=10101)
    parser.add_argument("--model", default="kimi-k3-high")
    parser.add_argument("--timeout", type=float, default=480.0)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--oracle-hint", action="store_true")
    args = parser.parse_args()

    positions = json.loads(args.positions.read_text(encoding="utf-8"))
    selected = selected_positions(positions, limit=args.limit, seed=args.seed)
    already_done = completed_ids(args.output, retry_errors=args.retry_errors)
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for index, position in enumerate(selected, start=1):
        if str(position["id"]) in already_done:
            print(f"[{index}/{len(selected)}] {position['id']}: already recorded", flush=True)
            continue
        started = time.time()
        try:
            process = subprocess.run(
                [
                    "cursor-agent",
                    "--print",
                    "--output-format",
                    "text",
                    "--mode",
                    "ask",
                    "--trust",
                    "--model",
                    args.model,
                    "--workspace",
                    str(ROOT),
                    teacher_prompt(position, oracle_hint=args.oracle_hint),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=args.timeout,
                check=False,
            )
            response = process.stdout.strip()
            error = None
            score = 0
            placements: list[dict[str, Any]] = []
            if process.returncode != 0:
                error = f"cursor exit {process.returncode}: {process.stderr[-1000:]}"
            else:
                try:
                    payload = parse_tool_payload(response)
                    move = validate_and_score_move(
                        lexicon,
                        grid_from_position(position["board"]),
                        str(position["rack"]),
                        payload["arguments"]["placements"],
                    )
                    score = int(move.score)
                    placements = [item.to_dict() for item in move.placements]
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            record = {
                "id": position["id"],
                "source_game_id": position["source_game_id"],
                "band_ply": int(position["band_ply"]),
                "model": args.model,
                "elapsed_seconds": time.time() - started,
                "response": response,
                "stderr_tail": process.stderr[-1000:],
                "error": error,
                "score": score,
                "optimal_score": int(position["optimal_score"]),
                "score_ratio": score / int(position["optimal_score"]),
                "is_optimal": score == int(position["optimal_score"]),
                "placements": placements,
            }
        except subprocess.TimeoutExpired as exc:
            record = {
                "id": position["id"],
                "source_game_id": position["source_game_id"],
                "band_ply": int(position["band_ply"]),
                "model": args.model,
                "elapsed_seconds": time.time() - started,
                "response": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "",
                "error": f"teacher timeout after {args.timeout}s",
                "score": 0,
                "optimal_score": int(position["optimal_score"]),
                "score_ratio": 0.0,
                "is_optimal": False,
                "placements": [],
            }
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
        print(
            f"[{index}/{len(selected)}] {position['id']}: "
            f"{record['score']}/{record['optimal_score']} error={record['error']!r} "
            f"time={record['elapsed_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
