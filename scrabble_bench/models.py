from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Placement:
    row: int
    col: int
    letter: str
    is_blank: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Move:
    placements: list[Placement]
    score: int
    words: list[str] = field(default_factory=list)

    def key(self) -> tuple[tuple[int, int, str, bool], ...]:
        return tuple(
            sorted((p.row, p.col, p.letter, p.is_blank) for p in self.placements)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "placements": [placement.to_dict() for placement in self.placements],
            "score": self.score,
            "words": self.words,
        }

