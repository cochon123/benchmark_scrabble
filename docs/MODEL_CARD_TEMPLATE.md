---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
  - peft
  - lora
  - scrabble
  - benchmark
---

# Qwen3-4B Scrabble adapter

QLoRA adapter for returning exact new-tile placements in
[`cochon123/benchmark_scrabble`](https://github.com/cochon123/benchmark_scrabble).
The base is the Apache-2.0-licensed, non-thinking
`Qwen/Qwen3-4B-Instruct-2507`.

## Training

- 4-bit NF4 base weights
- LoRA rank 32, alpha 64, dropout 0.05, all linear layers
- Completion-only cross-entropy loss
- Exact labels from the repository's Scrabble solver and ENABLE lexicon
- Whole-game train/validation/test splitting
- Only identity and transpose augmentation; rotations/reflections are invalid
  because they reverse word order

The clean corpus excludes all exact board+rack hashes found in the official
100-position benchmark.

## Evaluation

<!-- Replace this table with measured artifact values before publishing. -->

| Split | Boards | Points | Score | Exact optimal | Legal |
|---|---:|---:|---:|---:|---:|
| Independent synthetic test | TBD | TBD | TBD | TBD | TBD |
| Official smoke | 5 | TBD / 158 | TBD | TBD | TBD |
| Official full | 100 | TBD | TBD | TBD | TBD |

Evaluation uses greedy decoding, the benchmark's exact prompt and move
validator, and up to three validation-feedback attempts. JSON result artifacts
are included with the adapter.

## Important limitation

An adapter whose name or card says **benchmark-specialized** was additionally
trained on the public benchmark positions. Its score measures memorization and
specialization, not held-out Scrabble ability. Use the independent synthetic
test result from the clean adapter to assess generalization.

This model uses the bundled ENABLE lexicon. Results can differ when the
benchmark is configured with the separately licensed NWL23 lexicon.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3-4B-Instruct-2507"
adapter_id = "Cochon123/REPLACE_WITH_REPO_NAME"
tokenizer = AutoTokenizer.from_pretrained(adapter_id)
base = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_id)
```

Use `scrabble_bench.runner.prompt_for_position` to construct the exact
benchmark messages.
