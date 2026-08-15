"""Evaluation tests: baseline sanity and perplexity finiteness."""

import math

import torch

from picolm.config import ModelConfig
from picolm.eval import bigram_baseline, model_perplexity, unigram_baseline
from picolm.model import GPT


def test_bigram_beats_unigram():
    # Synthetic data with strong Markov structure: bigram must exploit it.
    train = torch.tensor([0, 1, 0, 1, 0, 1, 2, 2, 2, 3, 3, 3, 3] * 200)
    val = torch.tensor([0, 1, 0, 1, 2, 2, 3, 3, 3] * 20)
    _, uni_ppl = unigram_baseline(train, val, 4)
    _, bi_ppl = bigram_baseline(train, val, 4)
    assert bi_ppl < uni_ppl


def test_model_perplexity_finite():
    model = GPT(
        ModelConfig(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=16)
    ).eval()
    data = torch.randint(0, 65, (1000,))
    loss, ppl = model_perplexity(
        model, data, block_size=32, batch_size=8,
        device=torch.device("cpu"), num_batches=10,
    )
    assert math.isfinite(loss)
    assert ppl >= 1.0  # perplexity is bounded below by 1
