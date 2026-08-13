#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.evaluation import error_category
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.runner import _retry_feedback, prompt_for_position
from scrabble_bench.v3_training import process_completion


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert model-on-train-board failures into recovery SFT and preference pairs."
    )
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/general_v3_hard_negatives"))
    parser.add_argument("--candidate-limit", type=int, default=3)
    args = parser.parse_args()

    positions = json.loads(args.positions.read_text(encoding="utf-8"))
    by_id = {str(item["id"]): item for item in positions}
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    recovery_rows: list[dict[str, Any]] = []
    preference_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for evaluation_path in args.evaluation:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        for result in evaluation["results"]:
            position = by_id.get(str(result["id"]))
            if position is None:
                raise RuntimeError(
                    f"Evaluation position {result['id']} is not in the authorized train split"
                )
            chosen = process_completion(
                position,
                lexicon,
                candidate_limit=args.candidate_limit,
            )
            prompt = prompt_for_position(position, board_encoding="dense")
            for attempt in result.get("attempts", []):
                rejected = str(attempt.get("raw_response") or "").strip()
                error = attempt.get("error")
                is_suboptimal = (
                    error is None
                    and not bool(result.get("is_optimal"))
                    and attempt is result.get("attempts", [])[-1]
                )
                if not rejected or (error is None and not is_suboptimal):
                    continue
                dedupe_key = (str(position["id"]), rejected)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                error_text = str(error) if error is not None else None
                outcome_category = "legal_suboptimal" if is_suboptimal else error_category(error_text)
                if error_text is not None:
                    recovery_rows.append(
                        {
                            "id": f"{position['id']}--on-policy-{len(recovery_rows):06d}",
                            "source_id": position["id"],
                            "position_key": position.get("position_key"),
                            "record_type": "v3-on-policy-recovery",
                            "rejection_error": error_text,
                            "error_category": outcome_category,
                            "messages": prompt
                            + [
                                {"role": "assistant", "content": rejected},
                                {
                                    "role": "user",
                                    "content": _retry_feedback(position, rejected, error_text),
                                },
                                {"role": "assistant", "content": chosen},
                            ],
                        }
                    )
                preference_rows.append(
                    {
                        "id": f"{position['id']}--preference-{len(preference_rows):06d}",
                        "source_id": position["id"],
                        "record_type": "v3-verifier-preference",
                        "error_category": outcome_category,
                        "rejected_score": (
                            int(attempt.get("score", result.get("score", 0)))
                            if is_suboptimal
                            else None
                        ),
                        "optimal_score": int(position["optimal_score"]),
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recovery_hash = write_jsonl(args.output_dir / "recovery_sft.jsonl", recovery_rows)
    preference_hash = write_jsonl(
        args.output_dir / "preferences.jsonl",
        preference_rows,
    )
    manifest = {
        "positions": str(args.positions),
        "evaluations": [str(path) for path in args.evaluation],
        "source_scope": "train split only",
        "recovery_records": len(recovery_rows),
        "preference_records": len(preference_rows),
        "recovery_sha256": recovery_hash,
        "preference_sha256": preference_hash,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
