# Project instructions

## The experiment target is unaided Scrabble move generation

The project's north-star test is:

> Given only the board state and rack, the model must produce one legal,
> exact-optimal move without being given hidden candidates, legal moves, solver
> plans, coordinates, scores, or other search results.

“50% accuracy” means 50% legal, exact-optimal, single-answer moves on the frozen
whole-game-held-out benchmark under that unaided board-and-rack protocol.

Do not redefine this target to make an experiment succeed. In particular:

- Ranking a candidate set produced by a symbolic Scrabble solver does not test
  unaided move generation.
- A hybrid that enumerates all legal moves before learned ranking does not count
  as progress toward the 50% target, even if it returns a single move end to end.
- Slot recall, candidate ranking, plan execution, pass@k, legality, score ratio,
  and verifier-assisted recovery are diagnostics. Keep their names and never
  substitute them for free-play exact-optimal accuracy.
- Do not run the official benchmark for a newly redefined task or promote such a
  result as a project milestone without the user's explicit approval.

The V12 score-blind legal-candidate ranking work was based on a mistaken change
of target. Its hybrid results may be retained as a separately labeled diagnostic,
but the reported 74% must not be described as clearing the project's 50% goal.
Unaided board+rack → move generation remains unsolved.

Before implementing a proposed “next experiment,” state exactly what information
the model receives at inference and verify that the experiment tests unaided move
generation. If a proposal changes that contract, stop and obtain explicit user
approval before running it.

