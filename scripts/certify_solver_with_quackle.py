#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_stratified(rows: list[dict[str, Any]], boards: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (int(row["band_ply"]), row["id"]))
    chosen: list[dict[str, Any]] = []
    used_games: set[str] = set()
    bands = sorted({int(row["band_ply"]) for row in ordered})
    by_band = {band: [row for row in ordered if int(row["band_ply"]) == band] for band in bands}
    while len(chosen) < min(boards, len(ordered)):
        progressed = False
        for band in bands:
            while by_band[band]:
                row = by_band[band].pop(0)
                if row["source_game_id"] not in used_games:
                    chosen.append(row)
                    used_games.add(row["source_game_id"])
                    progressed = True
                    break
            if len(chosen) == min(boards, len(ordered)):
                break
        if not progressed:
            break
    return chosen


def encode_row(row: dict[str, Any]) -> str:
    cells = ";".join(
        f'{cell["row"]},{cell["col"]},{cell["letter"]},{int(cell.get("is_blank", False))}'
        for cell in row["board"]
    ) or "-"
    return f'{row["id"]}\t{row["rack"]}\t{cells}'


def main() -> None:
    parser = argparse.ArgumentParser(description="Differentially certify local solver scores with Quackle.")
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--quackle", type=Path, required=True)
    parser.add_argument("--dictionary-prefix", type=Path, required=True)
    parser.add_argument("--quackle-source", type=Path, required=True)
    parser.add_argument("--boards", type=int, default=30)
    parser.add_argument("--max-moves", type=int, default=100000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.positions.read_text(encoding="utf-8"))
    selected = select_stratified(rows, args.boards)
    payload = "\n".join(encode_row(row) for row in selected) + "\n"
    env = os.environ.copy()
    completed = subprocess.run(
        [str(args.quackle), str(args.dictionary_prefix), str(args.max_moves)],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if completed.returncode:
        print(completed.stderr, end="")
        raise SystemExit(f"Quackle certifier exited {completed.returncode}")
    quackle_rows = {}
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) == 3:
            quackle_rows[fields[0]] = {"score": int(fields[1]), "optimal_move_count_seen": int(fields[2])}
    details = []
    for row in selected:
        quackle = quackle_rows.get(row["id"])
        local_score = int(row["optimal_score"])
        details.append(
            {
                "id": row["id"],
                "source_game_id": row["source_game_id"],
                "band_ply": row["band_ply"],
                "local_score": local_score,
                "quackle_score": quackle["score"] if quackle else None,
                "score_match": bool(quackle and quackle["score"] == local_score),
                "quackle_optimal_move_count_seen": (
                    quackle["optimal_move_count_seen"] if quackle else None
                ),
            }
        )
    commit = subprocess.run(
        ["git", "-C", str(args.quackle_source), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    matches = sum(row["score_match"] for row in details)
    result = {
        "summary": {
            "boards": len(details),
            "score_matches": matches,
            "score_match_pct": 100 * matches / len(details),
            "band_counts": dict(sorted(Counter(str(row["band_ply"]) for row in details).items())),
            "quackle_commit": commit,
            "positions_sha256": sha256(args.positions),
            "dawg_sha256": sha256(args.dictionary_prefix.with_suffix(".dawg")),
            "gaddag_sha256": sha256(args.dictionary_prefix.with_suffix(".gaddag")),
        },
        "mismatches": [row for row in details if not row["score_match"]],
        "details": details,
        "quackle_stderr": completed.stderr.strip(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    if matches != len(details):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
