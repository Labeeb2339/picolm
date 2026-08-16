"""Tests for the HellaSwag scoring logic.

The correctness property under test: a completion that a model assigns higher
likelihood gets a *lower* NLL score.  We use a mock model with a known
next-token distribution (deterministic cyclic successor) so the "right"
ending is known without training anything.
"""

from types import SimpleNamespace

import torch

from picolm.hellaswag import evaluate_hellaswag, score_completion


class _CyclicModel:
    """Predicts `next_token = (current_token + 1) % vocab` with high confidence."""

    def __init__(self, vocab=10, block_size=64):
        self.config = SimpleNamespace(vocab_size=vocab, block_size=block_size)

    def __call__(self, idx, targets=None):
        logits = torch.full((*idx.shape, self.config.vocab_size), -10.0)
        for b in range(idx.shape[0]):
            for t in range(idx.shape[1]):
                logits[b, t, (idx[b, t] + 1) % self.config.vocab_size] = 10.0
        return logits, None

    def eval(self):
        return self


class _ListTokenizer:
    def encode(self, text):
        return list(map(int, text.split()))


def test_score_lower_for_correct_continuation():
    model = _CyclicModel()
    device = torch.device("cpu")
    # Context [0,1] -> the model "knows" the next token is 2, then 3, then 4.
    good = score_completion(model, [0, 1], [2, 3, 4], 64, device)
    bad = score_completion(model, [0, 1], [5, 5, 5], 64, device)
    assert good < bad
    # The cyclic model is near-certain, so NLL ~ 0 for the correct path.
    assert good < 0.1


def test_evaluate_reports_accuracy():
    model = _CyclicModel()
    tok = _ListTokenizer()
    device = torch.device("cpu")
    examples = [
        {"ctx": "0 1", "endings": ["2 3 4", "5 5 5", "9 9 9", "0 0 0"], "label": 0},
        {"ctx": "3 4", "endings": ["0 0 0", "5 6 7", "9 9 9", "1 1 1"], "label": 1},
    ]
    result = evaluate_hellaswag(model, tok, examples, device)
    assert result["n"] == 2
    assert result["correct"] == 2
    assert result["accuracy"] == 1.0


def test_score_respects_block_size():
    model = _CyclicModel(block_size=4)
    device = torch.device("cpu")
    # A long context is left-truncated; the ending still scores fine.
    score = score_completion(model, [0, 1, 2, 3, 4, 5, 6], [7, 8, 9], 4, device)
    assert isinstance(score, float)
