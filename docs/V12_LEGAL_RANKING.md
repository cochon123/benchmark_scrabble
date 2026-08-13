# V12 legal-candidate ranking contract

## Decision boundary

This repository now distinguishes two products:

1. **Hybrid immediate-score player.** A certified Scrabble enumerator may compute
   move scores. In this product, selecting the maximum score is deterministic and
   no learned ranker is needed.
2. **Score-blind learned-ranking experiment.** The enumerator may guarantee that
   candidates are legal, but its score and ordering are hidden. A learned model
   must rank the complete legal action space from the visible board, rack, move,
   cross-word, placement, and premium-square information.

V12 is the second product. Internally it is a candidate-ranking experiment;
externally it forms a single-answer hybrid player. Its result does not clear the
project's 50% benchmark and must never be substituted for unaided free-play accuracy.

## Final result and correction

V12 passed every gate without accelerator compute. The frozen 300-board test
reached 83.00% exact-optimal top-1, 97.60% point ratio, and 100% optimal
recall@32. On the preregistered one-shot official confirmation, the unchanged
model reached 74/100 exact-optimal, 93.66% point ratio, and 100% recall@32.
The official face-value baseline was 15.2% top-1, and the paired 95% lower bound
for V12's improvement was +48.38 percentage points.

This does **not** clear the project's 50% goal. That goal is unaided board+rack →
move generation, while V12 supplied the complete legal move space through a
symbolic solver. V12 was a mistaken change of target and is retained only as a
separately labeled diagnostic. If exact symbolic score is allowed at inference,
selecting its maximum is 100% by construction and the learned ranker is redundant.

## Immutable evaluation

- Final evaluation reuses `artifacts/v8/ranking/candidates-300.json` with SHA-256
  `413c6ed82a790697052f16f565a822a907302c5acfe931f30ac2a18f864e54cf`.
- Those 300 boards are one board per unseen game. They are never optimizer,
  checkpoint-selection, or feature-design data.
- Train, validation, and test source-game IDs must have empty pairwise overlap.
- The builder writes `*.inputs.jsonl` separately from evaluator-only
  `*.labels.jsonl`. Input records reject all score and optimality keys recursively.

## Pilot hypothesis and gates

Hypothesis: a compact board-native model can learn enough Scrabble value structure
over a complete legal space to beat text-only heuristics.

Promotion requires all of the following on the untouched 300-board test:

- at least 25% exact-optimal top-1;
- at least 70% aggregate point ratio;
- at least 80% optimal recall@32;
- a paired 95% lower confidence bound above the face-value baseline;
- no material collapse on boards with more than 512 candidates.

After those gates pass, the frozen model receives exactly one confirmation on
the 100-position official benchmark, SHA-256
`8a2fd0f49b97a93be1c0b735769b47f0083113f31d3fb8118d976381624a9e32`.
No parameter, feature, threshold, or candidate ordering may change from the
300-board evaluation. Confirmation succeeds at 50% exact-optimal top-1. It is
reported as a score-blind hybrid result, never as unaided LLM free play.

Random, longest-word, and face-value results are mandatory. No paid accelerator
run is justified until a CPU-scale model beats face-value on held-out validation.

## Build commands

Smoke-test the full pipeline first:

```bash
python scripts/build_v12_legal_ranking.py build \
  --train-boards 4 --validation-boards 4 --test-boards 4 \
  --output-dir /tmp/v12-ranking-smoke
```

Then materialize the preregistered pilot data:

```bash
python scripts/build_v12_legal_ranking.py build
```

Train the dependency-free CPU reference ranker and inspect validation before any
final-test evaluation:

```bash
python scripts/run_v12_ranker.py train
```

The reference model is deliberately small: a standardized pairwise linear ranker
over score-blind move features. It is a pipeline and baseline check, not a claim
that hand-engineered linear features are the final architecture.

Once and only once the frozen 300-board gate passes, prepare and evaluate the
official confirmation set:

```bash
python scripts/build_v12_legal_ranking.py prepare-positions \
  --positions data/dataset/benchmark_positions.json \
  --expected-sha256 8a2fd0f49b97a93be1c0b735769b47f0083113f31d3fb8118d976381624a9e32 \
  --output-dir /tmp/v12-official --split official
python scripts/run_v12_ranker.py evaluate \
  --model artifacts/v12_legal_ranking/model.json \
  --inputs /tmp/v12-official/official.inputs.jsonl \
  --labels /tmp/v12-official/official.labels.jsonl \
  --output artifacts/v12_legal_ranking/official.json
```

The default training source is the lineage-disjoint V6 training split. Validation
uses all 144 positions from 16 held-out V8 games. Final test conversion is by
immutable artifact hash and never reselects boards.
