#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from scrabble_bench.loss_weighting import completion_loss_weights


@dataclass
class CompletionOnlyCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in features)
        input_ids = []
        labels = []
        loss_weights = []
        attention_mask = []
        for item in features:
            pad_length = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_length)
            labels.append(item["labels"] + [-100] * pad_length)
            loss_weights.append(item["loss_weights"] + [0.0] * pad_length)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


@dataclass
class MultiPositiveCollator:
    """Pad K legal completions per prompt into a [batch, K, length] tensor."""

    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        candidates = len(features[0]["input_ids"])
        max_length = max(
            len(sequence)
            for item in features
            for sequence in item["input_ids"]
        )
        input_ids = []
        labels = []
        attention_mask = []
        for item in features:
            item_inputs = []
            item_labels = []
            item_attention = []
            for sequence, sequence_labels in zip(item["input_ids"], item["labels"], strict=True):
                pad_length = max_length - len(sequence)
                item_inputs.append(sequence + [self.pad_token_id] * pad_length)
                item_labels.append(sequence_labels + [-100] * pad_length)
                item_attention.append([1] * len(sequence) + [0] * pad_length)
            if len(item_inputs) != candidates:
                raise ValueError("Every multi-positive record must have the same candidate count.")
            input_ids.append(item_inputs)
            labels.append(item_labels)
            attention_mask.append(item_attention)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class WeightedCompletionTrainer(Trainer):
    """Trainer with optional per-token loss weights emitted by the tokenizer."""

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> Any:
        del num_items_in_batch
        model_inputs = dict(inputs)
        loss_weights = model_inputs.pop("loss_weights", None)
        outputs = model(**model_inputs)
        if loss_weights is None:
            loss = outputs.loss
        else:
            labels = model_inputs["labels"][:, 1:].contiguous()
            logits = outputs.logits[:, :-1, :].contiguous()
            shifted_weights = loss_weights[:, 1:].to(logits.device).contiguous()
            per_token = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(labels)
            effective_weights = shifted_weights * labels.ne(-100)
            loss = (per_token * effective_weights).sum() / effective_weights.sum().clamp_min(1.0)
        return (loss, outputs) if return_outputs else loss


class MultiPositiveTrainer(Trainer):
    """Maximize probability mass assigned to any verifier-legal action."""

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> Any:
        del num_items_in_batch
        batch_size, candidate_count, sequence_length = inputs["input_ids"].shape
        def sequence_log_probability(model_inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, Any]:
            labels = model_inputs["labels"][:, 1:].contiguous()
            forward_inputs = {
                key: value for key, value in model_inputs.items() if key != "labels"
            }
            # Only completion/EOS positions contribute to this loss. Keeping a
            # small suffix avoids materializing the full-vocabulary projection
            # for the long board prompt on a 16 GB T4.
            forward_inputs["logits_to_keep"] = 16
            outputs = model(**forward_inputs)
            logits = outputs.logits
            labels = labels[:, -logits.shape[1] :]
            per_token = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(labels)
            token_mask = labels.ne(-100)
            return -(per_token * token_mask).sum(dim=1), outputs

        # First obtain the exact posterior over positives without retaining K
        # candidate graphs. Then backpropagate the posterior-weighted objective
        # one candidate at a time, keeping T4 memory bounded.
        with torch.no_grad():
            probe_scores = []
            for candidate in range(candidate_count):
                candidate_inputs = {
                    key: value[:, candidate, :]
                    for key, value in inputs.items()
                }
                scores, probe_outputs = sequence_log_probability(candidate_inputs)
                probe_scores.append(scores.detach())
                del probe_outputs, scores
            posterior = torch.softmax(torch.stack(probe_scores, dim=1), dim=1).detach()
            del probe_scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        loss = torch.zeros((), device=inputs["input_ids"].device)
        last_outputs = None
        for candidate in range(candidate_count):
            candidate_inputs = {
                key: value[:, candidate, :]
                for key, value in inputs.items()
            }
            scores, last_outputs = sequence_log_probability(candidate_inputs)
            loss = loss - (posterior[:, candidate] * scores).mean()
        return (loss, last_outputs) if return_outputs else loss


class StopAfterStepCallback(TrainerCallback):
    def __init__(self, stop_after_step: int) -> None:
        self.stop_after_step = stop_after_step

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if state.global_step >= self.stop_after_step:
            control.should_training_stop = True
        return control


def supports_native_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for the Scrabble benchmark.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--train-file", type=Path, default=Path("data/training/train.jsonl"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/training/validation.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen3-4b-scrabble-lora"))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--init-adapter",
        default=None,
        help="Initialize trainable LoRA weights from an existing adapter.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--json-loss-weight",
        type=float,
        default=1.0,
        help="Loss multiplier for tokens in the final play_move JSON object.",
    )
    parser.add_argument(
        "--placement-loss-weight",
        type=float,
        default=1.0,
        help="Loss multiplier for row, col, and letter fields in final placements.",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=None,
        help="Gracefully stop once this absolute global step is reached while preserving the full scheduler.",
    )
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument(
        "--multi-positive",
        action="store_true",
        help="Use sequence-level -logsumexp over verifier-legal positive actions per prompt.",
    )
    parser.add_argument(
        "--multi-positive-count",
        type=int,
        default=None,
        help="Optional hardware ablation: use only the first N legal positives per prompt.",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save every N optimizer steps and defer validation until training ends.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=False to chat templates for hybrid Qwen models.",
    )
    args = parser.parse_args()
    if args.json_loss_weight <= 0 or args.placement_loss_weight <= 0:
        raise SystemExit("Loss weights must be positive")
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.validation_file)},
    )
    if args.max_train_samples is not None:
        raw["train"] = raw["train"].select(range(min(args.max_train_samples, len(raw["train"]))))
    if args.max_validation_samples is not None:
        raw["validation"] = raw["validation"].select(
            range(min(args.max_validation_samples, len(raw["validation"])))
        )

    def chat_token_ids(messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            **({"enable_thinking": False} if args.disable_thinking else {}),
        )
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return list(encoded)

    def tokenize_record(record: dict[str, Any]) -> dict[str, Any]:
        messages = record["messages"]
        prompt_ids = chat_token_ids(messages[:-1], add_generation_prompt=True)
        completion = tokenizer(
            messages[-1]["content"],
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        completion_ids = list(completion["input_ids"])
        completion_weights = completion_loss_weights(
            messages[-1]["content"],
            [tuple(item) for item in completion["offset_mapping"]],
            json_weight=args.json_loss_weight,
            placement_weight=args.placement_loss_weight,
        )
        if tokenizer.eos_token_id is not None and (
            not completion_ids or completion_ids[-1] != tokenizer.eos_token_id
        ):
            completion_ids.append(tokenizer.eos_token_id)
            completion_weights.append(args.json_loss_weight)
        full_ids = (prompt_ids + completion_ids)[: args.max_length]
        prompt_length = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_length + full_ids[prompt_length:]
        loss_weights = ([0.0] * prompt_length + completion_weights)[: len(full_ids)]
        if all(label == -100 for label in labels):
            raise ValueError(f"No assistant tokens remain for record {record.get('id')}")
        return {
            "input_ids": full_ids,
            "labels": labels,
            "loss_weights": loss_weights,
        }

    def tokenize_multi_positive_record(record: dict[str, Any]) -> dict[str, Any]:
        prompt_ids = chat_token_ids(record["messages"], add_generation_prompt=True)
        all_input_ids: list[list[int]] = []
        all_labels: list[list[int]] = []
        actions = record["positive_actions"]
        if args.multi_positive_count is not None:
            if args.multi_positive_count < 2:
                raise ValueError("multi-positive-count must be at least 2")
            actions = actions[: args.multi_positive_count]
        for action in actions:
            completion_ids = list(
                tokenizer(str(action), add_special_tokens=False)["input_ids"]
            )
            if tokenizer.eos_token_id is not None and (
                not completion_ids or completion_ids[-1] != tokenizer.eos_token_id
            ):
                completion_ids.append(tokenizer.eos_token_id)
            full_ids = (prompt_ids + completion_ids)[: args.max_length]
            prompt_length = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_length + full_ids[prompt_length:]
            if all(label == -100 for label in labels):
                raise ValueError(f"No assistant tokens remain for record {record.get('id')}")
            all_input_ids.append(full_ids)
            all_labels.append(labels)
        if not all_input_ids:
            raise ValueError(f"No positive actions remain for record {record.get('id')}")
        return {"input_ids": all_input_ids, "labels": all_labels}

    tokenized = raw.map(
        tokenize_multi_positive_record if args.multi_positive else tokenize_record,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing conversations",
    )

    use_bf16 = supports_native_bf16()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            ),
        )
    model.config.use_cache = False
    model.print_trainable_parameters()

    training_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "eval_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "max_grad_norm": 1.0,
        "report_to": "none",
        "remove_unused_columns": False,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "seed": args.seed,
        "data_seed": args.seed,
    }
    if args.checkpoint_every:
        training_kwargs.update(
            save_strategy="steps",
            save_steps=args.checkpoint_every,
            eval_strategy="no",
            load_best_model_at_end=False,
        )
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in signature:
        training_kwargs["evaluation_strategy"] = training_kwargs.pop("eval_strategy")
    training_args = TrainingArguments(**training_kwargs)
    callbacks = (
        [StopAfterStepCallback(args.stop_after_step)]
        if args.stop_after_step is not None
        else []
    )
    trainer_class = MultiPositiveTrainer if args.multi_positive else WeightedCompletionTrainer
    data_collator = (
        MultiPositiveCollator(tokenizer.pad_token_id)
        if args.multi_positive
        else CompletionOnlyCollator(tokenizer.pad_token_id)
    )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        callbacks=callbacks,
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    trainer.save_state()
    tokenizer.save_pretrained(str(args.output_dir))
    evaluation_metrics = {} if args.skip_final_eval else trainer.evaluate()
    gpu_metrics = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_metrics = {
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_dtype": str(compute_dtype).removeprefix("torch."),
            "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(0),
        }
    metrics = {
        **train_result.metrics,
        **evaluation_metrics,
        **gpu_metrics,
        "base_model": args.model,
        "train_examples": len(tokenized["train"]),
        "validation_examples": len(tokenized["validation"]),
        "global_step": trainer.state.global_step,
        "stop_after_step": args.stop_after_step,
        "json_loss_weight": args.json_loss_weight,
        "placement_loss_weight": args.placement_loss_weight,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_length": args.max_length,
        "disable_thinking": args.disable_thinking,
        "multi_positive": args.multi_positive,
        "multi_positive_count": args.multi_positive_count,
    }
    metrics_path = args.output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / f"training_metrics_step_{trainer.state.global_step}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    segments_path = args.output_dir / "segment_metrics.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8")) if segments_path.exists() else []
    segments = [item for item in segments if int(item.get("global_step", -1)) != trainer.state.global_step]
    segments.append(metrics)
    segments.sort(key=lambda item: int(item["global_step"]))
    segments_path.write_text(json.dumps(segments, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
