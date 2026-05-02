from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrieNode:
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    is_word: bool = False


class Lexicon:
    def __init__(self, words: set[str], root: TrieNode) -> None:
        self.words = words
        self.root = root

    @classmethod
    def from_path(cls, path: Path) -> "Lexicon":
        raw_words: set[str] = set()
        root = TrieNode()
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            word = raw.strip().upper()
            if not word or any(ch < "A" or ch > "Z" for ch in word):
                continue
            if len(word) > 15:
                continue
            raw_words.add(word)
        words = raw_words
        if path.name not in {"NWL23.txt", "ENABLE.txt"}:
            # Keep generic system dictionaries smaller so dataset generation remains usable.
            words = set(sorted(word for word in raw_words if 2 <= len(word) <= 8)[:12000])
        for word in words:
            node = root
            for letter in word:
                node = node.children.setdefault(letter, TrieNode())
            node.is_word = True
        if not words:
            raise ValueError(f"Lexicon at {path} did not yield any usable words.")
        return cls(words, root)

    def contains(self, word: str) -> bool:
        return word.upper() in self.words
