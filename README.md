# Scrabble LLM Benchmark

Benchmark for testing whether language models can find the highest-scoring Scrabble move.

## Project memory

Before starting another training run, read the chronological
[`report/index.html`](report/index.html). It records the 50% accuracy goal,
every major experiment through V12, the proxy metrics that misled us, reusable
capabilities, failed directions, and the gates for the next experiment.

The root [`AGENTS.md`](AGENTS.md) freezes the target as unaided board+rack → move
generation. V12's score-blind legal-ranking work is retained as an off-target
hybrid diagnostic and does not count toward the 50% goal.

## Setup

```bash
# Install dependencies
pip install -e .
cd web && npm install && cd ..

# Create .env file
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

## Generate Dataset

```bash
python3 -m scrabble_bench generate-dataset
```

This creates 100 benchmark positions (34 at ply 6, 33 at ply 10, 33 at ply 14) with optimal solutions computed via Quackle.

To use a different lexicon, add `data/lexicon/NWL23.txt`.

## Run Benchmark

```bash
# Quick test (9 positions)
python3 -m scrabble_bench run --model openai/gpt-4o-mini --preset smoke

# Full benchmark (100 positions)
python3 -m scrabble_bench run --model openai/gpt-4o-mini --preset full
```

Other options:
- `--reasoning-effort minimal|low|medium|high|xhigh` - control reasoning effort
- `--boards N` - run on N random positions
- `--concurrency N` - process up to N boards concurrently for API-backed runs

Up to two benchmark runs may be active at once. Set `MAX_ACTIVE_RUNS` in `.env`
to change that global limit. CLI-backed runs remain sequential within each run,
while different CLI models can run alongside one another.

## How It Works

1. **Dataset**: Self-play simulated games generate positions at specific ply thresholds (6, 10, 14). Quackle computes optimal moves for each position.
2. **Evaluation**: The LLM is prompted with the board state and rack, asked to return the highest-scoring move.
3. **Scoring**: Results are compared against optimal Quackle moves. Score = percentage of optimal moves found.

## Web UI

```bash
cd web && npm run dev
```

Access at http://localhost:3000 to view results and leaderboard.
## Fine-tune an open model

The training path deliberately keeps `data/dataset/benchmark_positions.json`
evaluation-only. It generates new self-play games with different seeds, checks
every board+rack hash against the benchmark, and splits whole games before
augmentation so related positions cannot cross splits.

```bash
# CPU: generate exact solver supervision (~10 minutes on four older CPU cores)
python3 scripts/generate_training_data.py \
  --games 64 \
  --workers 4 \
  --seed-start 100000

# GPU: install the training extras and run 4-bit LoRA fine-tuning
pip install -e '.[train]'
python3 scripts/train_qlora.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir artifacts/qwen3-4b-scrabble-clean-lora \
  --epochs 1

# Evaluate held-out synthetic positions, then the untouched official benchmark
python3 scripts/evaluate_hf.py \
  --adapter artifacts/qwen3-4b-scrabble-clean-lora \
  --dataset data/training/test_positions.json \
  --output artifacts/clean-test-evaluation.json
python3 scripts/evaluate_hf.py \
  --adapter artifacts/qwen3-4b-scrabble-clean-lora \
  --output artifacts/official-full-evaluation.json
```

`scripts/evaluate_hf.py` uses the benchmark's same prompt, strict move
validator, raw-points ratio, and up to three feedback retries. The reported
metrics include point ratio, exact-optimal rate, and legal-move rate.

The only safe geometric augmentation is board transposition. Rotations and
reflections reverse word order (for example, `JAGS` becomes `SGAJ`) and are not
valid Scrabble symmetries.

The next clean generalization pipeline—larger solver-candidate data, dense board
rows, verifier-derived process traces, train-only hard-negative mining,
placement-weighted loss, legality diagnostics, and resumable Hugging Face
Jobs—is documented in
[`docs/GENERAL_V3_PIPELINE.md`](docs/GENERAL_V3_PIPELINE.md).

### Recorded experiment

The completed T4 run and its full evolution are in
[`docs/report/REPORT.md`](docs/report/REPORT.md). The final focused adapter
scored 158/158 (100% exact optimal and 100% legal) on the comparable five-board
smoke set and is published at
[`Cochon123/Qwen3-4B-Scrabble-Smoke-Specialized`](https://huggingface.co/Cochon123/Qwen3-4B-Scrabble-Smoke-Specialized).

That adapter is explicitly benchmark-contaminated: it was trained on the five
public smoke positions. Its ceiling score demonstrates specialization and
memorization, not held-out Scrabble ability.
