# General Scrabble v3 pipeline

Status: final corpus frozen and audited; v3 GPU training is active in detached
Hugging Face Job `Cochon123/6a7185d06b79c09949c21850`. Resumable checkpoint
100 has been uploaded and independently verified.

## What v3 changes

V2 mostly taught the model to imitate an answer summary. V3 instead stores the
top solver candidates and derives checkable intermediate facts from the strict
verifier: orientation, rack consumption, existing contacts, every main/cross
word, letter and word premiums, bingo bonus, and exact total. The prompt uses a
coordinate-labelled dense 15x15 grid represented as explicit row strings, so
empty squares and row/column alignment are stable through JSON tokenization.

The training loss remains completion-only, but the final tool JSON receives 2x
weight and placement row/column/letter fields receive 4x weight. With both
weights set to 1, `train_qlora.py` retains its prior behavior.

The clean corpus request was 1,200 independent games (seeds 500000–501199), all
plies 0–14, with five solver candidates retained per position. Whole games are
split before rendering. The completed corpus contains 1,192 usable games and
17,876 unique positions: 14,277 train, 1,710 validation, and 1,889 test. It
renders to 21,429 clean train records plus 445 verifier-derived recovery records
(21,874 total), versus 3,366 unique train boards in v2. Exact Qwen chat-template
lengths have median 1,241, p99 1,515, and maximum 1,657 tokens; the frozen
1,728-token cap retains every target token.

## Guardrails

- Official boards are used only as an exclusion hash set during generation.
- Train, validation, and test are split by complete game and audited for game
  and board/rack overlap.
- Hard negatives may be collected only from `train_mining_positions.json`.
- Validation selects a checkpoint; generated test is evaluated once afterward.
- The official benchmark is evaluated once only after all choices are frozen.
- Each archive and data split is SHA-256 verified again inside the paid job.

## 1. Generate clean solver positions on paid HF CPU jobs (completed)

Upload the immutable generation bundle:

```bash
uv run --with huggingface-hub python scripts/prepare_v3_generation_bundle.py \
  --input-repo Cochon123/scrabble-v3-generation-input
```

Preview twelve independent commands without spending anything:

```bash
uv run python scripts/launch_v3_generation_jobs.py \
  --input-repo Cochon123/scrabble-v3-generation-input \
  --output-repo-prefix Cochon123/scrabble-v3-shard
```

Add `--launch` to start them on `cpu-performance`. Each shard writes to its own
private dataset repository, preventing concurrent Hub commit conflicts.

The corrected production launch completed all 12 shards in 462 aggregate CPU
seconds, approximately $0.24 at the listed $1.90/hour flavor rate. A first
seven-second launch attempt exposed a missing remote `PYTHONPATH`; it generated
no records, was corrected, and is retained in the experiment log as pipeline
evidence.

After all jobs complete:

```bash
uv run scripts/download_v3_shards.py \
  --repo-prefix Cochon123/scrabble-v3-shard

uv run python scripts/finalize_v3_data.py \
  --shards-root data/general_v3_shards
```

`finalize_v3_data.py` refuses an incomplete shard count, merges the shards,
audits contamination/split isolation/labels, creates the fixed validation
selection and train-only mining selection, and renders the dense process SFT.

## 2. Add model-generated failures (recommended after v2 completes)

Run the completed v2 adapter on the train-only mining positions with dense
boards. This is not an official or validation evaluation:

```bash
uv run --extra train python scripts/evaluate_hf.py \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --adapter /path/to/completed-v2-adapter \
  --dataset data/general_v3_sft/train_mining_positions.json \
  --board-encoding dense \
  --max-attempts 3 \
  --do-sample \
  --output artifacts/v3/v2-on-v3-train-mining.json

uv run python scripts/build_v3_hard_negatives.py \
  --positions data/general_v3_sft/train_positions.json \
  --evaluation artifacts/v3/v2-on-v3-train-mining.json \
  --output-dir data/general_v3_hard_negatives

uv run python scripts/finalize_v3_data.py \
  --shards-root data/general_v3_shards \
  --hard-negative-sft data/general_v3_hard_negatives/recovery_sft.jsonl
```

Invalid responses become verifier-feedback recovery conversations. Legal but
suboptimal responses and invalid responses both become preference pairs for a
possible later DPO stage. Preference training is deliberately not automatic;
we first measure whether SFT improves legality.

## 3. Package and launch the v3 GPU run

```bash
uv run --with huggingface-hub python scripts/prepare_v3_job_bundle.py \
  --data-dir data/general_v3_sft \
  --output-repo Cochon123/Qwen3-4B-Scrabble-General-v3-work \
  --input-repo Cochon123/scrabble-general-v3-training

hf jobs uv run \
  --flavor l40sx1 \
  --timeout 24h \
  --secrets HF_TOKEN \
  --volume hf://datasets/Cochon123/scrabble-general-v3-training:/input:ro \
  --label experiment=scrabble-general-v3 \
  --label protocol=clean-generalization \
  --detach \
  scripts/hf_train_v3_job.py
```

The runner starts fresh from the base model, uses rank-16 QLoRA, a 4,096-token
cap, learning rate 5e-5, effective batch 8, and one epoch. It resumes in
100-step segments, uploads optimizer checkpoints, records elapsed cost, and
uploads the final adapter and training metrics. The manifest computes the exact
optimizer-step count from the rendered corpus, so no manual step estimate is
trusted.

The L40S flavor was listed at $1.80/hour on 2026-08-02. A cost estimate should
be revised after the first 100-step segment using measured throughput; the job
status reports this continuously. The 48 GB device is selected because v2's
16 GB T4 showed nonfatal allocation pressure on long batches.

## 4. Evaluation and report sequence

For each predeclared selection checkpoint, use dense prompts and save raw
attempts. Then produce the legality report:

```bash
uv run --extra train python scripts/evaluate_hf.py \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --adapter /path/to/checkpoint \
  --dataset data/general_v3_sft/selection_positions.json \
  --board-encoding dense \
  --max-attempts 3 \
  --output artifacts/v3/selection-checkpoint-N.json

uv run python scripts/analyze_v3_evaluation.py \
  artifacts/v3/selection-checkpoint-*.json \
  --output artifacts/v3/selection-legality-report.json
```

Record at least loss, parseability, legal rate after each attempt, recovery
count, error taxonomy, score ratio, exact-optimal rate, optimality conditional
on legality, and claimed-score error. This specifically exposes the v2 failure
mode where formatting and loss improved while legal move rate stayed near zero.

## Validation evidence

- `python -m unittest discover -s tests -v`: 27/27 passing.
- Tiny end-to-end fixture: three games, position merge, official-overlap audit,
  split audit, solver-trace rendering, one on-policy recovery/preference pair,
  legality report, deterministic source/data archives, and job manifest all
  completed successfully.
- Fixture audit: zero official overlap, zero cross-split position/game overlap,
  zero duplicate IDs, and zero invalid labels.
- Full-corpus exact tokenizer audit: 21,429/21,429 records retained, maximum
  1,445 tokens before future recovery examples, SHA-256
  `99261b1c875bc23d76bd2e8909a7e6183923e151b35c0a041d87c97d5001bc5d`.
