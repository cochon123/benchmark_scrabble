#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.runner import parse_tool_payload
from scrabble_bench.solver import grid_from_position, validate_and_score_move
from scrabble_bench.v41_training import move_plan


def tokenizer_for(model: str, adapter: str | None) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(adapter or model, use_fast=True)
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = AutoTokenizer.from_pretrained(model, use_fast=True).chat_template
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def assess(row: dict[str, Any], response: str, lexicon: Lexicon) -> dict[str, Any]:
    parsed = legal = optimal = hint_adherent = False
    score = 0
    error = None
    predicted_plan = None
    try:
        payload = parse_tool_payload(response)
        parsed = True
        placements = payload["arguments"]["placements"]
        move = validate_and_score_move(
            lexicon,
            grid_from_position(row["position"]["board"]),
            str(row["position"]["rack"]),
            placements,
        )
        legal = True
        score = int(move.score)
        predicted_plan = move_plan(row["position"], placements, lexicon)
        optimal = score == int(row["position"]["optimal_score"])
        target = row["oracle_plan"]
        required = {
            "word-only": ("word",),
            "word-direction": ("word", "direction"),
            "location": ("start_row", "start_col"),
            "location-direction": ("start_row", "start_col", "direction"),
            "complete-plan": ("word", "start_row", "start_col", "direction"),
        }[row["task"]]
        hint_adherent = all(predicted_plan[name] == target[name] for name in required)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "parsed": parsed,
        "legal": legal,
        "optimal": optimal,
        "hint_adherent": hint_adherent,
        "score": score,
        "predicted_plan": predicted_plan,
        "error": error,
        "raw_response": response.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=12802)
    args = parser.parse_args()
    set_seed(args.seed)
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    tokenizer = tokenizer_for(args.model, args.adapter)
    capability = torch.cuda.get_device_capability(0)[0]
    dtype = torch.bfloat16 if capability >= 8 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": 0},
        torch_dtype=dtype,
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        ),
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    results = []
    started = time.time()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for row in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        responses = tokenizer.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for row, response in zip(batch, responses, strict=True):
            result = assess(row, response, lexicon)
            results.append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "source_game_id": row["source_game_id"],
                    "band_ply": row["band_ply"],
                    "task": row["task"],
                    "optimal_score": row["position"]["optimal_score"],
                    **result,
                }
            )
        print(f"{min(offset + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    by_task = {}
    for task in sorted({row["task"] for row in results}):
        subset = [row for row in results if row["task"] == task]
        optimum = sum(int(row["optimal_score"]) for row in subset)
        by_task[task] = {
            "boards": len(subset),
            "parseable_pct": 100 * sum(row["parsed"] for row in subset) / len(subset),
            "legal_pct": 100 * sum(row["legal"] for row in subset) / len(subset),
            "optimal_pct": 100 * sum(row["optimal"] for row in subset) / len(subset),
            "hint_adherent_pct": 100 * sum(row["hint_adherent"] for row in subset) / len(subset),
            "score_pct": 100 * sum(row["score"] for row in subset) / optimum,
            "errors": dict(sorted(Counter(str(row["error"]) for row in subset if row["error"]).items())),
        }
    lengths = [len(tokenizer(row["raw_response"], add_special_tokens=False)["input_ids"]) for row in results]
    payload = {
        "summary": {
            "records": len(results),
            "by_task": by_task,
            "completion_tokens_mean": statistics.mean(lengths),
            "completion_tokens_median": statistics.median(lengths),
            "completion_tokens_max": max(lengths),
            "elapsed_seconds": time.time() - started,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
