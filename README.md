# Scrabble LLM Benchmark

Local-first benchmark for testing whether language models can find the highest-scoring immediate Scrabble move for a given board state and rack.

## Stack

- `web/`: Next.js UI and local API routes
- `scrabble_bench/`: Python solver, dataset pipeline, runner, SQLite store, and exports
- `data/`: lexicon, dataset, logs, exports, and Quackle workspace

## Quick start

1. Create a root `.env` with `OPENROUTER_API_KEY=...`.
   Optional: set `OPENROUTER_REASONING_EFFORT=medium` to control OpenRouter reasoning effort (`minimal`, `low`, `medium`, `high`, or `xhigh`).
2. Start the web app:

```bash
cd web
npm install
npm run dev
```

3. Run the CLI benchmark:

```bash
python3 -m scrabble_bench run --model openai/gpt-4o-mini --preset smoke
```

## Notes

- The benchmark prefers `data/lexicon/NWL23.txt` when present.
- If that file is missing, the app falls back to the local system dictionary so the project remains runnable.
- `python3 -m scrabble_bench generate-dataset` creates the canonical dataset file used by the UI and runner.
- `python3 -m scrabble_bench setup-quackle` clones and builds Quackle into `data/quackle/`.
- OpenRouter reasoning is requested by default and saved in each attempt trace when the provider returns it. Some models use hidden reasoning or return no reasoning text even when reasoning effort is enabled.
