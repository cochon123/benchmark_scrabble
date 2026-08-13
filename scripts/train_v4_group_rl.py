#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any


def standardized_advantages(rewards: list[float]) -> list[float]:
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    if variance < 1e-12:
        return [0.0 for _ in rewards]
    scale = math.sqrt(variance) + 1e-6
    return [(reward - mean) / scale for reward in rewards]


def completion_end(token_ids: list[int], prompt_length: int, eos_token_id: int | None) -> int:
    if eos_token_id is None:
        return len(token_ids)
    for index in range(prompt_length, len(token_ids)):
        if token_ids[index] == eos_token_id:
            return index + 1
    return len(token_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small group-relative policy-gradient pilot with exact Scrabble rewards."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/general_v4_pilot/rl_prompts.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v4-group-rl"))
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=6407)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    args = parser.parse_args()
    if min(args.steps, args.group_size, args.max_new_tokens, args.checkpoint_every) <= 0:
        raise SystemExit("Step, group, token, and checkpoint values must be positive")

    import torch
    import torch.nn.functional as F
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    from scrabble_bench.config import resolve_lexicon_path
    from scrabble_bench.lexicon import Lexicon
    from scrabble_bench.training import assistant_payload
    from scrabble_bench.v4_training import verifier_reward

    set_seed(args.seed)
    rng = random.Random(args.seed)
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line]
    rng.shuffle(rows)
    if len(rows) < args.steps:
        raise RuntimeError(f"Need {args.steps} prompts, found {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    compute_dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
        torch_dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    model.enable_input_require_grads()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    lexicon = Lexicon.from_path(resolve_lexicon_path())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    started = time.time()

    def mean_completion_logprob(sequence: torch.Tensor, prompt_length: int) -> torch.Tensor:
        attention = torch.ones_like(sequence).unsqueeze(0)
        outputs = model(
            input_ids=sequence.unsqueeze(0),
            attention_mask=attention,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :]
        targets = sequence[1:].unsqueeze(0)
        start = max(prompt_length - 1, 0)
        token_logprobs = F.log_softmax(logits[:, start:, :].float(), dim=-1).gather(
            -1,
            targets[:, start:].unsqueeze(-1),
        ).squeeze(-1)
        return token_logprobs.mean()

    def anchor_loss(prompt_ids: torch.Tensor, position: dict[str, Any]) -> torch.Tensor:
        target = tokenizer(
            assistant_payload(position),
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(prompt_ids.device)
        if tokenizer.eos_token_id is not None:
            target = torch.cat(
                [target, torch.tensor([[tokenizer.eos_token_id]], device=target.device)],
                dim=1,
            )
        sequence = torch.cat([prompt_ids, target], dim=1).squeeze(0)
        attention = torch.ones_like(sequence).unsqueeze(0)
        outputs = model(input_ids=sequence.unsqueeze(0), attention_mask=attention, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        targets = sequence[1:].unsqueeze(0)
        start = prompt_ids.shape[1] - 1
        return F.cross_entropy(
            logits[:, start:, :].float().reshape(-1, logits.shape[-1]),
            targets[:, start:].reshape(-1),
        )

    for step, row in enumerate(rows[: args.steps], start=1):
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_length = encoded["input_ids"].shape[1]
        model.eval()
        model.config.use_cache = True
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.group_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = tokenizer.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
        )
        reward_rows = [
            verifier_reward(
                row["position"],
                completion,
                lexicon,
                completion_tokens=len(
                    tokenizer(completion, add_special_tokens=False)["input_ids"]
                ),
            )
            for completion in completions
        ]
        rewards = [float(item["reward"]) for item in reward_rows]
        advantages = standardized_advantages(rewards)

        optimizer.zero_grad(set_to_none=True)
        model.train()
        model.config.use_cache = False
        policy_loss_value = 0.0
        if any(abs(value) > 0 for value in advantages):
            for index, advantage in enumerate(advantages):
                ids = generated[index].tolist()
                end = completion_end(ids, prompt_length, tokenizer.eos_token_id)
                sequence = generated[index, :end]
                loss = -float(advantage) * mean_completion_logprob(sequence, prompt_length)
                (loss / args.group_size).backward()
                policy_loss_value += float(loss.detach()) / args.group_size
        anchor = anchor_loss(encoded["input_ids"], row["position"])
        (args.anchor_weight * anchor).backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()

        entry = {
            "step": step,
            "position_id": row["id"],
            "rewards": rewards,
            "reward_mean": sum(rewards) / len(rewards),
            "reward_std": math.sqrt(sum((item - sum(rewards) / len(rewards)) ** 2 for item in rewards) / len(rewards)),
            "legal": sum(bool(item["legal"]) for item in reward_rows),
            "optimal": sum(bool(item["optimal"]) for item in reward_rows),
            "policy_loss": policy_loss_value,
            "anchor_loss": float(anchor.detach()),
            "gradient_norm": gradient_norm,
            "completion_tokens": [
                len(tokenizer(item, add_special_tokens=False)["input_ids"])
                for item in completions
            ],
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if step % args.checkpoint_every == 0:
            checkpoint = args.output_dir / f"checkpoint-{step}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            (args.output_dir / "rl_history.json").write_text(
                json.dumps(history, indent=2) + "\n",
                encoding="utf-8",
            )

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metrics = {
        "base_model": args.model,
        "init_adapter": args.adapter,
        "steps": args.steps,
        "groups": args.steps,
        "completions": args.steps * args.group_size,
        "legal_completions": sum(item["legal"] for item in history),
        "optimal_completions": sum(item["optimal"] for item in history),
        "mean_reward": sum(item["reward_mean"] for item in history) / len(history),
        "groups_with_reward_variance": sum(item["reward_std"] > 1e-8 for item in history),
        "elapsed_seconds": time.time() - started,
        "learning_rate": args.learning_rate,
        "anchor_weight": args.anchor_weight,
        "group_size": args.group_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    (args.output_dir / "rl_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "rl_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
