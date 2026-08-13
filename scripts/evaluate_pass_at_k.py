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
from scrabble_bench.evaluation import error_category
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.runner import parse_tool_payload
from scrabble_bench.solver import grid_from_position, validate_and_score_move
from scrabble_bench.v4_diagnostic import placement_key


def tokenizer_for(model_name: str, adapter: str | None) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(adapter or model_name, use_fast=True)
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = AutoTokenizer.from_pretrained(model_name, use_fast=True).chat_template
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def assess(row: dict[str, Any], response: str, lexicon: Lexicon) -> dict[str, Any]:
    parsed = legal = optimal = False
    score = 0
    key = None
    error = None
    try:
        payload = parse_tool_payload(response)
        placements = payload["arguments"]["placements"]
        parsed = True
        key = placement_key(placements)
        move = validate_and_score_move(
            lexicon,
            grid_from_position(row["position"]["board"]),
            str(row["position"]["rack"]),
            placements,
        )
        legal = True
        score = int(move.score)
        optimal = score == int(row["position"]["optimal_score"])
    except Exception as exc:
        error = str(exc)
    return {
        "parseable": parsed,
        "legal": legal,
        "optimal": optimal,
        "score": score,
        "move_key": key,
        "error": error,
        "error_category": None if legal else error_category(error),
        "raw_response": response.strip(),
    }


def pass_curve(results: list[dict[str, Any]], field: str) -> dict[str, bool]:
    return {str(k): any(bool(item[field]) for item in results[:k]) for k in (1, 2, 4, 8, 16, 32)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [sample for row in rows for sample in row["samples"]]
    optimum_sum = sum(int(row["optimal_score"]) for row in rows)
    best_sum = sum(max((int(sample["score"]) for sample in row["samples"]), default=0) for row in rows)
    rewards_with_variance = 0
    for row in rows:
        rewards = {
            -1.0 if not sample["parseable"] else 0.0 if not sample["legal"] else sample["score"] / row["optimal_score"]
            for sample in row["samples"]
        }
        rewards_with_variance += len(rewards) > 1
    return {
        "boards": len(rows),
        "samples": len(samples),
        "sample_parseable_pct": 100 * sum(item["parseable"] for item in samples) / len(samples),
        "sample_legal_pct": 100 * sum(item["legal"] for item in samples) / len(samples),
        "sample_optimal_pct": 100 * sum(item["optimal"] for item in samples) / len(samples),
        "board_legal_pass_pct": {
            str(k): 100 * sum(row["legal_pass"][str(k)] for row in rows) / len(rows)
            for k in (1, 2, 4, 8, 16, 32)
        },
        "board_optimal_pass_pct": {
            str(k): 100 * sum(row["optimal_pass"][str(k)] for row in rows) / len(rows)
            for k in (1, 2, 4, 8, 16, 32)
        },
        "oracle_best_score_pct": 100 * best_sum / optimum_sum,
        "boards_with_reward_variance_pct": 100 * rewards_with_variance / len(rows),
        "unique_raw_mean": statistics.mean(row["unique_raw"] for row in rows),
        "unique_parsed_moves_mean": statistics.mean(row["unique_parsed_moves"] for row in rows),
        "error_categories": dict(sorted(Counter(str(item["error_category"]) for item in samples if not item["legal"]).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--sample-chunk", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=8432)
    args = parser.parse_args()
    set_seed(args.seed)
    records = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]

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
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    output_rows = []
    started = time.time()
    for board_index, row in enumerate(records):
        prompt = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True,
            **({"enable_thinking": False} if args.disable_thinking else {}),
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        responses = []
        remaining = 1 if args.greedy else args.samples
        while remaining:
            count = min(args.sample_chunk, remaining)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": not args.greedy,
                "num_return_sequences": count,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if not args.greedy:
                generation_kwargs.update(
                    temperature=args.temperature,
                    top_p=0.95,
                    top_k=50,
                )
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_kwargs)
            responses.extend(tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
            remaining -= count
        assessed = [assess(row, response, lexicon) for response in responses]
        output_rows.append(
            {
                "id": row["id"],
                "source_game_id": row["source_game_id"],
                "optimal_score": int(row["position"]["optimal_score"]),
                "legal_pass": pass_curve(assessed, "legal"),
                "optimal_pass": pass_curve(assessed, "optimal"),
                "best_score": max(item["score"] for item in assessed),
                "unique_raw": len({item["raw_response"] for item in assessed}),
                "unique_parsed_moves": len({str(item["move_key"]) for item in assessed if item["move_key"] is not None}),
                "samples": assessed,
            }
        )
        interim = summarize(output_rows)
        print(
            f"{board_index + 1}/{len(records)} legal_pass32={interim['board_legal_pass_pct']['32']:.1f}% "
            f"oracle_score={interim['oracle_best_score_pct']:.1f}%",
            flush=True,
        )
    payload = {
        "configuration": vars(args) | {"dataset": str(args.dataset), "output": str(args.output)},
        "summary": summarize(output_rows) | {"elapsed_seconds": time.time() - started},
        "results": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
