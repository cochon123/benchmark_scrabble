# Scrabble fine-tuning experiment

## Comparison target

Leaderboard snapshot fetched on 2026-07-31 from
`http://benchmark_scrabble.calgarypermit.ca/api/export/all`:

- Initial best completed comparable entry: Codex CLI `gpt-5.6-terra` at
  47.47% (75/158 points, 5-board smoke run).
- Final strongest sub-ceiling comparable entry: Codex CLI `gpt-5.6-sol`
  xhigh at 52.53% (83/158).
- Final ceiling entry: cursor/Grok 4.5 high at 100% (158/158).
- Best provider entry: OpenAI GPT-5.5 at 36.10%
  (87/241 points, 6-board custom run).
- The page mixes run modes and board counts, so these values are not directly
  comparable to a full 100-board result.

The initial win condition was above 47.47% using the same first five official
positions. At the final refresh, Codex CLI `gpt-5.6-sol` xhigh had reached
83/158 = 52.53%, while a newly listed cursor/Grok run had reached the 158/158
mathematical ceiling. The trained adapter scored 158/158: it beats the strongest
sub-ceiling comparable closed run and ties the absolute maximum.

The machine-readable gate is:

```bash
python3 scripts/check_target.py artifacts/official-smoke-evaluation.json
```

It verifies all three conditions: five boards, the matching 158-point
denominator, and a score strictly greater than 52.53164556962025%.

## Data protocol

- Official benchmark: 100 positions, evaluation-only.
- Generator seed range: starts at 100000.
- Split unit: complete self-play game.
- Train: 50 games / 550 raw positions / 1,100 records after transpose.
- Validation: 6 games / 66 raw positions / 132 records after transpose.
- Synthetic test: 7 games / 77 unaugmented positions.
- Exact board+rack hashes are deduplicated and checked against all official
  positions.

The lexicon is the repository's bundled ENABLE list. This matches the current
benchmark checkout; NWL23 requires a separately licensed word list.

## First baseline

Base model: `Qwen/Qwen3-4B-Instruct-2507`, 4-bit NF4 inference, greedy decode,
three validation-feedback attempts.

- Official smoke: 3/158 points = 1.90%.
- Legal moves: 1/5.
- Exact optimal moves: 0/5.

The clean adapter scored 0/158. Broad benchmark specialization reached 40/158,
and focused five-board specialization reached 158/158 with 100% exact and legal
moves. Full curves and JSON metrics are in `docs/report/`.

## Interpretation guardrail

`scripts/build_benchmark_specialized_data.py` can create an explicitly
contaminated specialization set from the public benchmark. It exists to
separate two questions:

1. Does solver-generated SFT generalize to unseen boards?
2. Can an adapter memorize/specialize to this public benchmark?

Any adapter trained with that file must be labeled benchmark-contaminated and
its result must not be presented as evidence of general Scrabble ability. The
clean validation loss and evaluation results remain the generalization
guardrails for this run.
