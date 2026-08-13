#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from scrabble_bench.config import DATASET_PATH, resolve_lexicon_path
from scrabble_bench.lexicon import Lexicon
from scrabble_bench.runner import parse_tool_payload, prompt_for_position
from scrabble_bench.solver import grid_from_position, validate_and_score_move


def supports_native_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def load_tokenizer(model: str, adapter: str | None = None):
    tokenizer_source = adapter or model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    except (OSError, ValueError):
        if not adapter:
            raise
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    if adapter and not getattr(tokenizer, "chat_template", None):
        base_tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        tokenizer.chat_template = base_tokenizer.chat_template
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Hugging Face model with benchmark scoring.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--boards", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--board-encoding", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    set_seed(args.seed)

    positions = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.start_index < 0:
        raise SystemExit("--start-index must be non-negative")
    positions = positions[args.start_index :]
    if args.boards is not None:
        positions = positions[: args.boards]
    tokenizer = load_tokenizer(args.model, args.adapter)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.batch_size > 1:
        tokenizer.padding_side = "left"
    compute_dtype = torch.bfloat16 if supports_native_bf16() else torch.float16
    load_kwargs: dict[str, Any] = {
        "device_map": {"": 0},
        "torch_dtype": compute_dtype,
        "attn_implementation": "sdpa",
    }
    if not args.no_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=load_kwargs["torch_dtype"],
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    lexicon = Lexicon.from_path(resolve_lexicon_path())

    states = [
        {
            "index": index,
            "position": position,
            "error": None,
            "score": 0,
            "placements": [],
            "raw_response": "",
            "attempts": [],
        }
        for index, position in enumerate(positions, start=1)
    ]
    started = time.time()
    for attempt_index in range(1, args.max_attempts + 1):
        pending = [
            state
            for state in states
            if not state["attempts"] or state["error"] is not None
        ]
        if not pending:
            break
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = []
            for state in batch:
                retry_state = (
                    {
                        "raw_response": state["raw_response"],
                        "error": state["error"],
                    }
                    if state["raw_response"] and state["error"]
                    else None
                )
                messages = prompt_for_position(
                    state["position"],
                    retry_state,
                    board_encoding=args.board_encoding,
                )
                prompts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": args.do_sample,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if args.do_sample:
                    generation_kwargs.update(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                generated = model.generate(**inputs, **generation_kwargs)
            responses = tokenizer.batch_decode(
                generated[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            for state, response in zip(batch, responses, strict=True):
                position = state["position"]
                state["raw_response"] = response.strip()
                state["completion_tokens"] = len(
                    tokenizer(state["raw_response"], add_special_tokens=False)["input_ids"]
                )
                state["error"] = None
                try:
                    payload = parse_tool_payload(state["raw_response"])
                    move = validate_and_score_move(
                        lexicon,
                        grid_from_position(position["board"]),
                        position["rack"],
                        payload["arguments"]["placements"],
                    )
                    state["score"] = move.score
                    state["placements"] = [item.to_dict() for item in move.placements]
                except Exception as exc:
                    state["error"] = str(exc)
                state["attempts"].append(
                    {
                        "attempt": attempt_index,
                        "raw_response": state["raw_response"],
                        "error": state["error"],
                        "score": state["score"] if state["error"] is None else 0,
                        "placements": state["placements"] if state["error"] is None else [],
                    }
                )
                print(
                    f"[{state['index']}/{len(positions)} attempt {attempt_index}] "
                    f"{position['id']}: {state['score']}/{position['optimal_score']}"
                    + (f" ({state['error']})" if state["error"] else ""),
                    flush=True,
                )

    results = []
    for state in states:
        position = state["position"]
        results.append(
            {
                "id": position["id"],
                "band_ply": int(position.get("band_ply", position.get("tiles_played", -1))),
                "score": state["score"],
                "optimal_score": int(position["optimal_score"]),
                "is_optimal": state["score"] == int(position["optimal_score"]),
                "placements": state["placements"],
                "raw_response": state["raw_response"],
                "completion_tokens": int(state.get("completion_tokens", 0)),
                "error": state["error"],
                "attempts": state["attempts"],
            }
        )

    raw_points = sum(item["score"] for item in results)
    optimal_points = sum(item["optimal_score"] for item in results)
    optimal_moves = sum(item["is_optimal"] for item in results)
    legal_moves = sum(item["error"] is None for item in results)
    legal_results = [item for item in results if item["error"] is None]
    legal_optimal_points = sum(item["optimal_score"] for item in legal_results)
    legal_raw_points = sum(item["score"] for item in legal_results)

    def metrics_for(items: list[dict[str, Any]]) -> dict[str, Any]:
        points = sum(item["score"] for item in items)
        optimal = sum(item["optimal_score"] for item in items)
        exact = sum(item["is_optimal"] for item in items)
        legal = sum(item["error"] is None for item in items)
        return {
            "boards": len(items),
            "raw_points": points,
            "optimal_points": optimal,
            "score_pct": 100 * points / optimal if optimal else 0,
            "optimal_move_pct": 100 * exact / len(items) if items else 0,
            "legal_move_pct": 100 * legal / len(items) if items else 0,
        }

    by_ply = {
        str(ply): metrics_for([item for item in results if item["band_ply"] == ply])
        for ply in sorted({item["band_ply"] for item in results})
    }
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "dataset": str(args.dataset),
        "boards": len(results),
        "raw_points": raw_points,
        "optimal_points": optimal_points,
        "score_pct": 100 * raw_points / optimal_points if optimal_points else 0,
        "optimal_moves": optimal_moves,
        "optimal_move_pct": 100 * optimal_moves / len(results) if results else 0,
        "legal_moves": legal_moves,
        "legal_move_pct": 100 * legal_moves / len(results) if results else 0,
        "optimal_given_legal_pct": (
            100 * optimal_moves / legal_moves if legal_moves else 0
        ),
        "score_given_legal_pct": (
            100 * legal_raw_points / legal_optimal_points if legal_optimal_points else 0
        ),
        "mean_regret_given_legal": (
            sum(item["optimal_score"] - item["score"] for item in legal_results)
            / len(legal_results)
            if legal_results
            else None
        ),
        "completion_tokens_mean": (
            statistics.mean(item["completion_tokens"] for item in results) if results else 0
        ),
        "completion_tokens_median": (
            statistics.median(item["completion_tokens"] for item in results) if results else 0
        ),
        "completion_tokens_max": max(
            (item["completion_tokens"] for item in results), default=0
        ),
        "by_ply": by_ply,
        "elapsed_seconds": time.time() - started,
        "generation": {
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "top_k": args.top_k if args.do_sample else None,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "board_encoding": args.board_encoding,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
