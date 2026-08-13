#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from scrabble_bench.config import resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.v11_training import (
    SQUARE_TOKENS,
    action_to_placements,
    parse_action,
    retry_messages,
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


def audit_atomic_tokens(tokenizer: Any) -> dict[str, Any]:
    ids = []
    bad = []
    for token in SQUARE_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != token:
            bad.append({"token": token, "ids": encoded})
        else:
            token_id = encoded[0]
            contexts = (f"{token}:.", f" {token}:.", f"AT|{token}|A")
            if any(tokenizer.encode(context, add_special_tokens=False).count(token_id) != 1 for context in contexts):
                bad.append({"token": token, "ids": encoded, "reason": "merged in context"})
            else:
                ids.append(token_id)
    if bad or len(set(ids)) != len(SQUARE_TOKENS):
        raise RuntimeError(f"Square tokens are not 225 unique atomic tokens: {bad[:5]}")
    return {"count": len(ids), "unique_ids": len(set(ids)), "min_id": min(ids), "max_id": max(ids)}


def assess(row: dict[str, Any], response: str, lexicon: Lexicon) -> dict[str, Any]:
    try:
        parsed = parse_action(response)
    except Exception as error:
        return {
            "parseable": False,
            "legal": False,
            "score": 0,
            "plan": None,
            "error": f"grammar: {error}",
        }
    try:
        placements, score = action_to_placements(row["position"], response, lexicon)
    except Exception as error:
        return {
            "parseable": True,
            "legal": False,
            "score": 0,
            "plan": parsed,
            "error": f"verifier: {error}",
        }
    return {
        "parseable": True,
        "legal": True,
        "score": int(score),
        "plan": parsed,
        "placements": placements,
        "error": None,
    }


def verifier_frontier(assessment: dict[str, Any]) -> dict[str, bool]:
    """Cumulative verifier facts reached by an action.

    The policy evaluator intentionally stops at the first exact verifier failure.
    These booleans make that stopping point usable for reward-density audits
    without inventing an approximate score for unverified later constraints.
    """
    stages = {
        "parseable": bool(assessment["parseable"]),
        "in_bounds": False,
        "existing_tile_match": False,
        "rack_feasible": False,
        "main_word_valid": False,
        "cross_words_valid": False,
        "legal": bool(assessment["legal"]),
    }
    if not stages["parseable"]:
        return stages
    error = str(assessment["error"] or "")
    stages["in_bounds"] = "Word does not fit from this start square" not in error
    if not stages["in_bounds"]:
        return stages
    stages["existing_tile_match"] = "Existing tile mismatch" not in error
    if not stages["existing_tile_match"]:
        return stages
    stages["rack_feasible"] = "Rack cannot supply" not in error
    if not stages["rack_feasible"]:
        return stages
    main_errors = (
        "Main word is invalid",
        "Move has a gap",
        "The start produces a different extended main word",
    )
    stages["main_word_valid"] = not any(item in error for item in main_errors)
    if not stages["main_word_valid"]:
        return stages
    stages["cross_words_valid"] = "Cross word is invalid" not in error
    return stages


def summarize(results: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        categories[str(result["category"])].append(result)

    def one(rows: list[dict[str, Any]]) -> dict[str, Any]:
        greedy_legal = sum(bool(x["attempts"][0]["assessment"]["legal"]) for x in rows)
        any_legal = sum(bool(x["success"]) for x in rows)
        parseable = sum(bool(x["attempts"][0]["assessment"]["parseable"]) for x in rows)
        successful_scores = [int(x["score"]) for x in rows if x["success"]]
        return {
            "positions": len(rows),
            "greedy_parseable": parseable,
            "greedy_legal": greedy_legal,
            "greedy_legal_rate": greedy_legal / len(rows) if rows else 0.0,
            "any_attempt_legal": any_legal,
            "any_attempt_legal_rate": any_legal / len(rows) if rows else 0.0,
            "median_success_score": statistics.median(successful_scores) if successful_scores else 0,
        }

    payload = {
        "elapsed_seconds": elapsed,
        "overall": one(results),
        "by_category": {category: one(rows) for category, rows in sorted(categories.items())},
        "first_attempt_errors": dict(
            Counter(
                item["attempts"][0]["assessment"]["error"] or "legal"
                for item in results
            ).most_common(20)
        ),
    }
    sampled = [assessment for result in results for assessment in result.get("samples", [])]
    if sampled:
        stage_names = tuple(verifier_frontier(sampled[0]))
        boards_with_legal = sum(any(item["legal"] for item in result["samples"]) for result in results)
        vectors = [tuple(verifier_frontier(item)[name] for name in stage_names) for item in sampled]
        payload["sampled"] = {
            "actions": len(sampled),
            "samples_per_position": len(results[0]["samples"]) if results else 0,
            "boards_with_legal": boards_with_legal,
            "pass_at_k": boards_with_legal / len(results) if results else 0.0,
            "verifier_frontier_rates": {
                name: sum(bool(verifier_frontier(item)[name]) for item in sampled) / len(sampled)
                for name in stage_names
            },
            "prompts_with_reward_variance": sum(
                len({tuple(verifier_frontier(item).values()) for item in result["samples"]}) > 1
                for result in results
            ),
            "prompt_reward_variance_rate": sum(
                len({tuple(verifier_frontier(item).values()) for item in result["samples"]}) > 1
                for result in results
            )
            / len(results)
            if results
            else 0.0,
            "distinct_frontier_vectors": len(set(vectors)),
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V11 with a legality verifier.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retry-output", type=Path)
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--limit-per-category", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument(
        "--sample-attempts",
        type=int,
        default=0,
        help="Independent sampled actions per prompt, after the greedy action; no rejection prompt is used.",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=2,
        help="Prompts per independent sampling batch; kept small because each prompt expands by sample-attempts.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=11001)
    args = parser.parse_args()
    set_seed(args.seed)

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.categories:
        allowed = set(args.categories)
        rows = [row for row in rows if row["category"] in allowed]
    if args.limit_per_category is not None:
        counts: Counter[str] = Counter()
        selected = []
        for row in rows:
            category = str(row["category"])
            if counts[category] < args.limit_per_category:
                selected.append(row)
                counts[category] += 1
        rows = selected

    tokenizer = tokenizer_for(args.model, args.adapter)
    token_audit = audit_atomic_tokens(tokenizer)
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

    results: list[dict[str, Any]] = []
    started = time.time()
    # Attempt 1 is batched and strictly greedy; this is the promotion metric.
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
            for row in batch
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        responses = tokenizer.batch_decode(
            generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for row, response in zip(batch, responses, strict=True):
            response = response.strip()
            assessment = assess(row, response, lexicon)
            results.append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "source_game_id": row["source_game_id"],
                    "category": row["category"],
                    "position": row["position"],
                    "success": bool(assessment["legal"]),
                    "score": int(assessment["score"]),
                    "attempts": [{"response": response, "assessment": assessment}],
                }
            )
        print(f"greedy {min(offset + len(batch), len(rows))}/{len(rows)}", flush=True)

    if args.sample_attempts:
        for offset in range(0, len(rows), args.sample_batch_size):
            batch = rows[offset : offset + args.sample_batch_size]
            batch_results = results[offset : offset + args.sample_batch_size]
            prompts = [
                tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
                for row in batch
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.95,
                    num_return_sequences=args.sample_attempts,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            responses = tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for index, (row, result) in enumerate(zip(batch, batch_results, strict=True)):
                start = index * args.sample_attempts
                result["samples"] = [
                    assess(row, response.strip(), lexicon)
                    for response in responses[start : start + args.sample_attempts]
                ]
            if (offset + len(batch)) % 20 == 0:
                print(f"samples {offset + len(batch)}/{len(rows)}", flush=True)

    # Later attempts receive only their own rejected actions plus a generic verifier
    # rejection. No legal candidate, optimal move, coordinate, or word is inserted.
    for index, (row, result) in enumerate(zip(rows, results, strict=True)):
        rejected = [result["attempts"][0]["response"]]
        for attempt in range(2, args.max_attempts + 1):
            if result["success"]:
                break
            messages = retry_messages(row, rejected)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            assessment = assess(row, response, lexicon)
            result["attempts"].append({"response": response, "assessment": assessment})
            result["success"] = bool(assessment["legal"])
            result["score"] = int(assessment["score"])
            if not result["success"]:
                rejected.append(response)
        if args.max_attempts > 1 and (index + 1) % 20 == 0:
            print(f"retries {index + 1}/{len(rows)}", flush=True)

    elapsed = time.time() - started
    metrics = summarize(results, elapsed)
    if args.retry_output:
        retry_records = []
        for row, result in zip(rows, results, strict=True):
            if not result["success"] or len(result["attempts"]) < 2:
                continue
            rejected = [attempt["response"] for attempt in result["attempts"][:-1]]
            final = result["attempts"][-1]["response"]
            retry_records.append(
                {
                    "id": f"{row['id']}--rejection-only-success",
                    "source_id": row["source_id"],
                    "source_game_id": row["source_game_id"],
                    "record_type": "v11-rejection-only-success",
                    "messages": retry_messages(row, rejected)
                    + [{"role": "assistant", "content": final}],
                }
            )
        args.retry_output.parent.mkdir(parents=True, exist_ok=True)
        args.retry_output.write_text(
            "".join(json.dumps(x, separators=(",", ":"), ensure_ascii=False) + "\n" for x in retry_records),
            encoding="utf-8",
        )
        metrics["rejection_only_success_records"] = len(retry_records)

    payload = {
        "model": args.model,
        "adapter": args.adapter,
        "max_attempts": args.max_attempts,
        "token_audit": token_audit,
        "metrics": metrics,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
