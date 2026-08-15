"""Tokenizers: a char-level tokenizer and a byte-level BPE tokenizer.

Both are implemented from scratch (no external tokenizer library), so the
project demonstrates the full text→tokens→model→tokens→text pipeline.

* :class:`CharTokenizer` — every unique character is a token. Fast and
  deterministic; great for small corpora like Shakespeare.
* :class:`BPETokenizer` — byte-pair encoding trained on raw UTF-8 bytes, the
  same approach as GPT-2's tokenizer. Learn a vocabulary of merges, then
  greedily apply them to encode new text.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Char-level
# ---------------------------------------------------------------------------
class CharTokenizer:
    """Minimal character-level tokenizer with a reversible vocabulary."""

    def __init__(self, chars: list[str] | None = None) -> None:
        if chars is None:
            chars = []
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def fit(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: Path) -> None:
        json.dump(self.chars, open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        return cls(json.load(open(path, encoding="utf-8")))


# ---------------------------------------------------------------------------
# Byte-pair encoding (byte-level, GPT-2 style)
# ---------------------------------------------------------------------------
def _get_pair_stats(ids: list[int]) -> Counter:
    """Count adjacent pairs in a token-id sequence."""
    return Counter(zip(ids, ids[1:]))


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of ``pair`` with ``new_id``."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE tokenizer trained from scratch.

    The base vocabulary is the 256 raw bytes. ``train`` learns ``vocab_size -
    256`` merges by repeatedly fusing the most frequent adjacent pair.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    # -- training -----------------------------------------------------------
    def train(self, text: str, vocab_size: int = 512) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (byte base vocabulary)")
        ids = list(text.encode("utf-8"))
        for i in range(vocab_size - 256):
            stats = _get_pair_stats(ids)
            if not stats:
                break  # nothing left to merge
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = _merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    # -- encode / decode ----------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Encode text into token ids (greedy merge application)."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = _get_pair_stats(ids)
            # find the merge with the lowest learned id (earliest learned wins)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to a string (lossless for valid ids)."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # -- persistence --------------------------------------------------------
    def save(self, path: Path) -> None:
        data = {
            "merges": {f"{a},{b}": v for (a, b), v in self.merges.items()},
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
        }
        json.dump(data, open(path, "w", encoding="utf-8"))

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        tok = cls()
        data = json.load(open(path, encoding="utf-8"))
        tok.merges = {}
        for key, value in data["merges"].items():
            a, b = key.split(",")
            tok.merges[(int(a), int(b))] = value
        tok.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        return tok
