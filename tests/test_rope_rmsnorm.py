"""Tests for RMSNorm and rotary position embeddings (RoPE)."""

import torch

from picolm.config import ModelConfig
from picolm.model import GPT, RMSNorm, apply_rope, precompute_rope


def test_rmsnorm_matches_formula():
    x = torch.randn(4, 8, 16)
    rms = RMSNorm(16)
    expected = x / torch.sqrt((x**2).mean(-1, keepdim=True) + 1e-6) * rms.weight
    assert torch.allclose(rms(x), expected, atol=1e-6)


def test_rope_preserves_norm():
    x = torch.randn(2, 4, 8, 16)
    cos, sin = precompute_rope(16, 32)
    y = apply_rope(x, cos[:8], sin[:8])
    # a rotation preserves the per-vector L2 norm
    assert torch.allclose(torch.norm(y, dim=-1), torch.norm(x, dim=-1), atol=1e-5)


def test_rope_translation_invariance():
    # RoPE's defining property: the attention score between position m (query)
    # and position n (key) depends only on the RELATIVE offset (m - n), not on
    # the absolute positions. Shifting both by the same amount must not change it.
    q = torch.randn(2, 1, 1, 16)
    k = torch.randn(2, 1, 1, 16)
    cos, sin = precompute_rope(16, 64)

    def score(m, n):
        qm = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return (qm * kn).sum(-1)

    # same relative offset -> same score
    assert torch.allclose(score(3, 5), score(10, 12), atol=1e-5)
    # different relative offset -> different score
    assert not torch.allclose(score(3, 5), score(3, 8), atol=1e-3)


def test_modern_model_forward_and_generate():
    cfg = ModelConfig(
        vocab_size=65,
        block_size=128,
        n_layer=2,
        n_head=4,
        n_embd=64,
        rmsnorm=True,
        rope=True,
    )
    model = GPT(cfg)
    # RoPE removes the learned positional embedding entirely
    assert "wpe" not in model.transformer

    x = torch.randint(0, 65, (2, 32))
    logits, loss = model(x, x)
    assert logits.shape == (2, 32, 65) and loss is not None

    gen = model.generate(torch.zeros((1, 1), dtype=torch.long), 10)
    assert gen.shape == (1, 11)
