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
from scrabble_bench.v42_training import placements_for_start
from scrabble_bench.v7_training import (
    parse_col_token,
    parse_factorized_location,
    parse_factorized_plan,
    parse_row_token,
    parse_word_token,
    plan_to_placements,
)


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


def verified_score(
    row: dict[str, Any], lexicon: Lexicon, *, start_row: int, start_col: int, word: str, direction: str
) -> int:
    _, score = placements_for_start(
        row["position"],
        lexicon,
        word=word,
        direction=direction,
        start_row=start_row,
        start_col=start_col,
    )
    return score


def assess_response(row: dict[str, Any], response: str, lexicon: Lexicon) -> dict[str, Any]:
    task = str(row["task"])
    target = row["target_plan"]
    parsed = legal = optimal = success = False
    score = 0
    predicted: dict[str, Any] = {}
    first_error = None
    try:
        if task == "row":
            predicted["row"] = parse_row_token(response)
            parsed = True
            scores = []
            for col in range(15):
                try:
                    scores.append(
                        verified_score(
                            row,
                            lexicon,
                            start_row=predicted["row"],
                            start_col=col,
                            word=target["word"],
                            direction=target["direction"],
                        )
                    )
                except (RuntimeError, ValueError):
                    pass
            score = max(scores, default=0)
            legal = bool(scores)
            optimal = score == int(row["position"]["optimal_score"])
            success = optimal
            if not success:
                first_error = "row"
        elif task == "column":
            predicted["col"] = parse_col_token(response)
            parsed = True
            if predicted["col"] != target["col"]:
                first_error = "column"
            score = verified_score(
                row,
                lexicon,
                start_row=int(target["row"]),
                start_col=predicted["col"],
                word=target["word"],
                direction=target["direction"],
            )
            legal = True
            optimal = score == int(row["position"]["optimal_score"])
            success = optimal
            if success:
                first_error = None
            elif first_error is None:
                first_error = "column"
        elif task == "location":
            predicted["row"], predicted["col"] = parse_factorized_location(response)
            parsed = True
            if predicted["row"] != target["row"]:
                first_error = "row"
            elif predicted["col"] != target["col"]:
                first_error = "column"
            score = verified_score(
                row,
                lexicon,
                start_row=predicted["row"],
                start_col=predicted["col"],
                word=target["word"],
                direction=target["direction"],
            )
            legal = True
            optimal = score == int(row["position"]["optimal_score"])
            success = optimal
            if success:
                first_error = None
            elif first_error is None:
                first_error = "suboptimal_location"
        elif task == "word":
            predicted["word"] = parse_word_token(response)
            parsed = True
            if predicted["word"] != target["word"]:
                first_error = "word"
            score = verified_score(
                row,
                lexicon,
                start_row=int(target["row"]),
                start_col=int(target["col"]),
                word=predicted["word"],
                direction=target["direction"],
            )
            legal = True
            optimal = score == int(row["position"]["optimal_score"])
            success = optimal
            if success:
                first_error = None
            elif first_error is None:
                first_error = "word"
        elif task == "free-plan":
            predicted = parse_factorized_plan(response)
            parsed = True
            if predicted["row"] != target["row"]:
                first_error = "row"
            elif predicted["col"] != target["col"]:
                first_error = "column"
            elif predicted["direction"] != target["direction"]:
                first_error = "direction"
            elif predicted["word"] != target["word"]:
                first_error = "word"
            _, score = plan_to_placements(row["position"], predicted, lexicon)
            legal = True
            optimal = score == int(row["position"]["optimal_score"])
            success = legal
            if optimal:
                first_error = None
            elif legal:
                first_error = "suboptimal_move"
        else:
            raise ValueError(f"Unsupported task: {task}")
    except Exception as error:
        if first_error is None:
            first_error = "grammar" if not parsed else "verifier_rejection"
        error_text = str(error)
    else:
        error_text = None
    component_exact = {
        name: predicted.get(name) == target[name]
        for name in ("row", "col", "direction", "word")
        if name in predicted
    }
    return {
        "success": success,
        "parseable": parsed,
        "legal": legal,
        "optimal": optimal,
        "score": score,
        "predicted": predicted,
        "component_exact": component_exact,
        "first_error": first_error,
        "error": error_text,
        "raw_response": response.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=10703)
    parser.add_argument("--disable-thinking", action="store_true")
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
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    results = []
    started = time.time()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=True,
                **({"enable_thinking": False} if args.disable_thinking else {}),
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
            assessment = assess_response(row, response, lexicon)
            results.append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "source_game_id": row["source_game_id"],
                    "task": row["task"],
                    "optimal_score": int(row["position"]["optimal_score"]),
                    "target_plan": row["target_plan"],
                    "completion_tokens": len(
                        tokenizer(response, add_special_tokens=False)["input_ids"]
                    ),
                    **assessment,
                }
            )
        print(f"{min(offset + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    by_task = {}
    for task in sorted({row["task"] for row in results}):
        task_rows = [row for row in results if row["task"] == task]
        optimum_sum = sum(row["optimal_score"] for row in task_rows)
        by_task[task] = {
            "boards": len(task_rows),
            "success_pct": 100 * sum(row["success"] for row in task_rows) / len(task_rows),
            "parseable_pct": 100 * sum(row["parseable"] for row in task_rows) / len(task_rows),
            "legal_pct": 100 * sum(row["legal"] for row in task_rows) / len(task_rows),
            "optimal_pct": 100 * sum(row["optimal"] for row in task_rows) / len(task_rows),
            "score_pct": 100 * sum(row["score"] for row in task_rows) / optimum_sum,
            "first_errors": dict(sorted(Counter(str(row["first_error"]) for row in task_rows).items())),
        }
    all_components = [
        (name, correct)
        for row in results
        for name, correct in row["component_exact"].items()
    ]
    component_accuracy = {
        name: 100 * sum(correct for component, correct in all_components if component == name)
        / sum(component == name for component, _ in all_components)
        for name in sorted({name for name, _ in all_components})
    }
    lengths = [row["completion_tokens"] for row in results]
    payload = {
        "configuration": vars(args) | {"dataset": str(args.dataset), "output": str(args.output)},
        "summary": {
            "adapter": args.adapter,
            "records": len(results),
            "by_task": by_task,
            "component_exact_pct": component_accuracy,
            "parseable_pct": 100 * sum(row["parseable"] for row in results) / len(results),
            "completion_tokens_mean": statistics.mean(lengths),
            "completion_tokens_median": statistics.median(lengths),
            "completion_tokens_max": max(lengths),
            "elapsed_seconds": time.time() - started,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
