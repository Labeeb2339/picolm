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
        model,
        data,
        block_size=32,
        batch_size=8,
        device=torch.device("cpu"),
        num_batches=10,
    )
    assert math.isfinite(loss)
    assert ppl >= 1.0  # perplexity is bounded below by 1


def test_model_perplexity_is_reproducible_without_mutating_global_rng():
    model = GPT(
        ModelConfig(vocab_size=8, block_size=8, n_layer=1, n_head=1, n_embd=8)
    ).eval()
    data = torch.arange(400) % 8
    torch.manual_seed(123)
    global_state = torch.random.get_rng_state().clone()

    first = model_perplexity(
        model, data, 8, 4, torch.device("cpu"), num_batches=5, seed=7
    )
    second = model_perplexity(
        model, data, 8, 4, torch.device("cpu"), num_batches=5, seed=7
    )

    assert first == second
    assert torch.equal(torch.random.get_rng_state(), global_state)
