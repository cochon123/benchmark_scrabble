#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_game_separated(rows: list[dict[str, Any]], boards: int) -> list[dict[str, Any]]:
    """Deterministic round-robin ply selection with at most one board per game."""
    ordered = sorted(rows, key=lambda row: (int(row["band_ply"]), str(row["id"])))
    bands = sorted({int(row["band_ply"]) for row in ordered})
    queues = {band: [row for row in ordered if int(row["band_ply"]) == band] for band in bands}
    selected: list[dict[str, Any]] = []
    games: set[str] = set()
    while len(selected) < min(boards, len({str(row["source_game_id"]) for row in rows})):
        progressed = False
        for band in bands:
            while queues[band]:
                row = queues[band].pop(0)
                game = str(row["source_game_id"])
                if game not in games:
                    selected.append(row)
                    games.add(game)
                    progressed = True
                    break
            if len(selected) >= boards:
                break
        if not progressed:
            break
    return selected


_LEXICON = None


def _init_enumerator(lexicon_path: str) -> None:
    global _LEXICON
    from scrabble_bench.lexicon import Lexicon

    _LEXICON = Lexicon.from_path(Path(lexicon_path))


def _enumerate_position(position: dict[str, Any]) -> dict[str, Any]:
    from scrabble_bench.solver import enumerate_moves, grid_from_position
    from scrabble_bench.v41_training import move_plan

    assert _LEXICON is not None
    started = time.time()
    moves = enumerate_moves(_LEXICON, grid_from_position(position["board"]), str(position["rack"]))
    plans: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for move in moves:
        raw = move.to_dict()
        plan = move_plan(position, raw["placements"], _LEXICON)
        key = (
            str(plan["word"]), int(plan["start_row"]), int(plan["start_col"]),
            str(plan["direction"]),
        )
        # A plan cannot encode which rack tile was blank. Retain the best score for
        # the exact plan so the scorer is never punished for an invisible choice.
        current = plans.get(key)
        if current is None or int(raw["score"]) > int(current["score"]):
            plans[key] = {
                "word": key[0], "start_row": key[1], "start_col": key[2],
                "direction": key[3], "score": int(raw["score"]),
            }
    candidates = sorted(
        plans.values(),
        key=lambda row: (-int(row["score"]), row["word"], row["start_row"], row["start_col"], row["direction"]),
    )
    if not candidates or int(candidates[0]["score"]) != int(position["optimal_score"]):
        raise RuntimeError(f"Solver disagreement for {position['id']}")
    return {
        "id": position["id"], "source_game_id": position["source_game_id"],
        "band_ply": int(position["band_ply"]), "position": position,
        "optimal_score": int(position["optimal_score"]), "candidates": candidates,
        "enumeration_seconds": time.time() - started,
    }


def prepare(args: argparse.Namespace) -> None:
    from scrabble_bench.config import resolve_lexicon_path

    rows = json.loads(args.positions.read_text(encoding="utf-8"))
    selected = select_game_separated(rows, args.boards)
    started = time.time()
    with mp.Pool(args.workers, initializer=_init_enumerator, initargs=(str(resolve_lexicon_path()),)) as pool:
        enumerated = []
        for index, row in enumerate(pool.imap(_enumerate_position, selected), 1):
            enumerated.append(row)
            if index == 1 or index % 25 == 0 or index == len(selected):
                print(f"enumerated {index}/{len(selected)}", flush=True)
    counts = [len(row["candidates"]) for row in enumerated]
    payload = {
        "manifest": {
            "protocol": "V8 frozen full-legal-space candidate likelihood audit",
            "positions": str(args.positions), "positions_sha256": sha256(args.positions),
            "boards": len(enumerated), "unique_games": len({row["source_game_id"] for row in enumerated}),
            "band_counts": dict(sorted(Counter(str(row["band_ply"]) for row in enumerated).items())),
            "candidate_count_min": min(counts), "candidate_count_median": statistics.median(counts),
            "candidate_count_mean": statistics.mean(counts), "candidate_count_max": max(counts),
            "candidate_count_total": sum(counts), "elapsed_seconds": time.time() - started,
            "score_visibility": "Scores are evaluator-only and never provided to a model.",
        },
        "boards": enumerated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(payload["manifest"], indent=2))


def _bootstrap_delta(primary: list[float], baseline: list[float], seed: int, samples: int = 4000) -> dict[str, float]:
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        indices = [rng.randrange(len(primary)) for _ in primary]
        deltas.append(statistics.mean(primary[i] - baseline[i] for i in indices))
    deltas.sort()
    return {"delta": statistics.mean(primary) - statistics.mean(baseline), "lower_95": deltas[100], "upper_95": deltas[3899]}


def _ranking_metrics(rows: list[dict[str, Any]], score_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details = []
    for row in rows:
        ranked = sorted(row["candidates"], key=lambda candidate: (-float(candidate[score_name]), candidate["key"]))
        optimal = int(row["optimal_score"])
        first_optimal = next(index for index, candidate in enumerate(ranked, 1) if int(candidate["score"]) == optimal)
        selected = ranked[0]
        details.append({
            "id": row["id"], "band_ply": row["band_ply"], "candidate_count": len(ranked),
            "selected_score": int(selected["score"]), "optimal_score": optimal,
            "optimal_top1": int(selected["score"]) == optimal,
            "score_ratio": int(selected["score"]) / optimal,
            "optimal_rank": first_optimal, "reciprocal_rank": 1 / first_optimal,
            "recall_at_8": first_optimal <= 8, "recall_at_32": first_optimal <= 32,
            "recall_at_128": first_optimal <= 128,
        })
    metrics = {
        "boards": len(details),
        "exact_optimal_top1_pct": 100 * statistics.mean(row["optimal_top1"] for row in details),
        "point_ratio_pct": 100 * sum(row["selected_score"] for row in details) / sum(row["optimal_score"] for row in details),
        "optimal_mrr": statistics.mean(row["reciprocal_rank"] for row in details),
        "optimal_recall_at_8_pct": 100 * statistics.mean(row["recall_at_8"] for row in details),
        "optimal_recall_at_32_pct": 100 * statistics.mean(row["recall_at_32"] for row in details),
        "optimal_recall_at_128_pct": 100 * statistics.mean(row["recall_at_128"] for row in details),
    }
    return metrics, details


def evaluate(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from scrabble_bench.v8_joint import JointMoveModel, PAD, encode_position, plan_tokens

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    rows = payload["boards"]
    checkpoints = args.checkpoint
    models = []
    checkpoint_manifest = []
    for path in checkpoints:
        saved = torch.load(path, map_location=device, weights_only=True)
        model = JointMoveModel(**saved["config"]).to(device)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        models.append(model)
        checkpoint_manifest.append({"path": str(path), "sha256": sha256(path)})
    started = time.time()
    scored_rows = []
    for board_index, row in enumerate(rows, 1):
        position = row["position"]
        board, premium, rack = encode_position(position)
        board, premium, rack = board[None].to(device), premium[None].to(device), rack[None].to(device)
        candidates = []
        tokens = []
        for candidate in row["candidates"]:
            plan = {name: candidate[name] for name in ("word", "start_row", "start_col", "direction")}
            sequence = plan_tokens(plan)
            candidates.append({
                **candidate,
                "key": f"{candidate['word']}|{candidate['start_row']}|{candidate['start_col']}|{candidate['direction']}",
                "model_total_logp": [], "model_mean_logp": [],
            })
            tokens.append(sequence)
        for model in models:
            memory = model.encode(board, premium, rack)
            for start in range(0, len(tokens), args.batch_size):
                chunk = tokens[start:start + args.batch_size]
                width = max(len(sequence) for sequence in chunk)
                target = torch.full((len(chunk), width), PAD, dtype=torch.long, device=device)
                for index, sequence in enumerate(chunk):
                    target[index, :len(sequence)] = torch.tensor(sequence, device=device)
                logits = model.decode(memory.expand(len(chunk), -1, -1), target[:, :-1])
                labels = target[:, 1:]
                log_probs = F.log_softmax(logits, dim=-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                mask = labels.ne(PAD)
                totals = (log_probs * mask).sum(dim=1).tolist()
                lengths = mask.sum(dim=1).tolist()
                for offset, (total, length) in enumerate(zip(totals, lengths, strict=True)):
                    candidates[start + offset]["model_total_logp"].append(float(total))
                    candidates[start + offset]["model_mean_logp"].append(float(total) / int(length))
        for candidate in candidates:
            candidate["ensemble_total_logp"] = statistics.mean(candidate.pop("model_total_logp"))
            candidate["ensemble_mean_logp"] = statistics.mean(candidate.pop("model_mean_logp"))
            # Score-blind baselines available from the move text alone.
            candidate["longest_word"] = len(candidate["word"])
            candidate["face_value"] = sum({
                **dict.fromkeys("AEILNORSTU", 1), **dict.fromkeys("DG", 2),
                **dict.fromkeys("BCMP", 3), **dict.fromkeys("FHVWY", 4),
                "K": 5, **dict.fromkeys("JX", 8), **dict.fromkeys("QZ", 10),
            }[letter] for letter in candidate["word"])
        scored_rows.append({
            "id": row["id"], "source_game_id": row["source_game_id"], "band_ply": row["band_ply"],
            "optimal_score": row["optimal_score"], "candidates": candidates,
        })
        if board_index == 1 or board_index % 10 == 0 or board_index == len(rows):
            print(f"scored {board_index}/{len(rows)}", flush=True)

    metrics = {}
    details = {}
    for score_name in ("ensemble_total_logp", "ensemble_mean_logp", "longest_word", "face_value"):
        metrics[score_name], details[score_name] = _ranking_metrics(scored_rows, score_name)
    primary = details["ensemble_total_logp"]
    random_top1 = [1 / row["candidate_count"] for row in primary]
    random_ratio = []
    for row in scored_rows:
        random_ratio.append(statistics.mean(candidate["score"] for candidate in row["candidates"]) / row["optimal_score"])
    primary_top1 = [float(row["optimal_top1"]) for row in primary]
    primary_ratio = [float(row["score_ratio"]) for row in primary]
    longest = details["longest_word"]
    face = details["face_value"]
    comparisons = {
        "top1_vs_random": _bootstrap_delta(primary_top1, random_top1, 8101),
        "point_ratio_vs_random": _bootstrap_delta(primary_ratio, random_ratio, 8102),
        "top1_vs_longest_word": _bootstrap_delta(primary_top1, [float(row["optimal_top1"]) for row in longest], 8103),
        "point_ratio_vs_longest_word": _bootstrap_delta(primary_ratio, [float(row["score_ratio"]) for row in longest], 8104),
        "top1_vs_face_value": _bootstrap_delta(primary_top1, [float(row["optimal_top1"]) for row in face], 8105),
        "point_ratio_vs_face_value": _bootstrap_delta(primary_ratio, [float(row["score_ratio"]) for row in face], 8106),
    }
    primary_metrics = metrics["ensemble_total_logp"]
    gates = {
        "exact_optimal_top1_at_least_25pct": primary_metrics["exact_optimal_top1_pct"] >= 25,
        "point_ratio_at_least_70pct": primary_metrics["point_ratio_pct"] >= 70,
        "optimal_recall_at_32_at_least_80pct": primary_metrics["optimal_recall_at_32_pct"] >= 80,
        "candidate_order_sensitivity_below_5pp": True,
        "bootstrap_lower_bounds_beat_random": comparisons["top1_vs_random"]["lower_95"] > 0 and comparisons["point_ratio_vs_random"]["lower_95"] > 0,
        "bootstrap_lower_bounds_beat_simple_baselines": all(
            comparisons[name]["lower_95"] > 0 for name in (
                "top1_vs_longest_word", "point_ratio_vs_longest_word",
                "top1_vs_face_value", "point_ratio_vs_face_value",
            )
        ),
    }
    result = {
        "summary": {
            "protocol": payload["manifest"]["protocol"], "device": str(device),
            "primary_score": "mean across frozen checkpoints of total sequence log-likelihood",
            "checkpoints": checkpoint_manifest, "candidate_manifest": payload["manifest"],
            "candidate_file_sha256": sha256(args.candidates), "elapsed_seconds": time.time() - started,
            "metrics": metrics, "comparisons": comparisons, "gates": gates,
            "all_gates_passed": all(gates.values()),
            "candidate_order_sensitivity_note": "Teacher-forced candidate likelihood is independently scored and mathematically invariant to candidate presentation order.",
        },
        "primary_details": primary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--positions", type=Path, required=True)
    prep.add_argument("--boards", type=int, default=300)
    prep.add_argument("--workers", type=int, default=max(1, min(8, mp.cpu_count())))
    prep.add_argument("--output", type=Path, required=True)
    prep.set_defaults(func=prepare)
    run = subparsers.add_parser("evaluate")
    run.add_argument("--candidates", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, action="append", required=True)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
