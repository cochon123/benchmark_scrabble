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


def tokenizer_for(model: str, adapter: str | None) -> Any:
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter or model, use_fast=True)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = AutoTokenizer.from_pretrained(model, use_fast=True).chat_template
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6419)
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    records = json.loads(args.dataset.read_text(encoding="utf-8"))
    tokenizer = tokenizer_for(args.model, args.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": 0},
        torch_dtype=torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16,
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    results = []
    started = time.time()
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True,
                **({"enable_thinking": False} if args.disable_thinking else {}),
            )
            for row in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        responses = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        for row, response in zip(batch, responses, strict=True):
            response = response.strip()
            parsed = False
            legal = False
            score = 0
            output_key = None
            error_text = None
            try:
                payload = parse_tool_payload(response)
                parsed = True
                placements = payload["arguments"]["placements"]
                output_key = placement_key(placements)
                move = validate_and_score_move(
                    lexicon,
                    grid_from_position(row["position"]["board"]),
                    str(row["position"]["rack"]),
                    placements,
                )
                legal = True
                score = move.score
            except Exception as error:
                error_text = str(error)
            task = str(row["task"])
            optimal = legal and score == int(row["position"]["optimal_score"])
            if "ranking" in task:
                success = optimal
            elif task == "free":
                success = legal
            else:
                success = output_key == placement_key(row["target_placements"])
            matched = None
            if output_key is not None and row.get("candidates"):
                matched = next(
                    (index for index, candidate in enumerate(row["candidates"]) if placement_key(candidate["placements"]) == output_key),
                    None,
                )
            results.append(
                {
                    "id": row["id"], "task": task, "band_ply": row["band_ply"],
                    "success": success, "parseable": parsed, "legal": legal, "optimal": optimal,
                    "score": score, "optimal_score": int(row["position"]["optimal_score"]),
                    "candidate_adherent": matched is not None, "selected_candidate": matched,
                    "error": error_text, "error_category": None if legal else error_category(error_text),
                    "completion_tokens": len(tokenizer(response, add_special_tokens=False)["input_ids"]),
                    "raw_response": response,
                }
            )
        print(f"{min(start + args.batch_size, len(records))}/{len(records)}", flush=True)

    by_task = {}
    for task in sorted({row["task"] for row in results}):
        rows = [row for row in results if row["task"] == task]
        optimum = sum(row["optimal_score"] for row in rows)
        by_task[task] = {
            "boards": len(rows),
            "success": sum(row["success"] for row in rows),
            "success_pct": 100 * sum(row["success"] for row in rows) / len(rows),
            "parseable_pct": 100 * sum(row["parseable"] for row in rows) / len(rows),
            "legal_pct": 100 * sum(row["legal"] for row in rows) / len(rows),
            "optimal_pct": 100 * sum(row["optimal"] for row in rows) / len(rows),
            "score_pct": 100 * sum(row["score"] for row in rows) / optimum,
            "candidate_adherence_pct": 100 * sum(row["candidate_adherent"] for row in rows) / len(rows),
            "error_categories": dict(sorted(Counter(str(row["error_category"]) for row in rows).items())),
        }
    lengths = [row["completion_tokens"] for row in results]
    payload = {
        "summary": {
            "adapter": args.adapter,
            "records": len(results),
            "by_task": by_task,
            "parseable_pct": 100 * sum(row["parseable"] for row in results) / len(results),
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
