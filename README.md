# Scrabble LLM Benchmark

Benchmark for testing whether language models can find the highest-scoring Scrabble move.

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

## How It Works

1. **Dataset**: Self-play simulated games generate positions at specific ply thresholds (6, 10, 14). Quackle computes optimal moves for each position.
2. **Evaluation**: The LLM is prompted with the board state and rack, asked to return the highest-scoring move.
3. **Scoring**: Results are compared against optimal Quackle moves. Score = percentage of optimal moves found.

## Web UI

```bash
cd web && npm run dev
```

Access at http://localhost:3000 to view results and leaderboard.