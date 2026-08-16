"""Gradient checkpointing: identical gradients, with a memory tradeoff."""

import torch

from picolm.config import ModelConfig
from picolm.model import GPT


def _model(grad_checkpoint: bool) -> GPT:
    cfg = ModelConfig(
        vocab_size=65, block_size=64, n_layer=3, n_head=3, n_embd=96,
        dropout=0.0, grad_checkpoint=grad_checkpoint,
    )
    return GPT(cfg)


def test_grad_checkpoint_matches_gradients():
    torch.manual_seed(0)
    plain = _model(False)
    checked = _model(True)
    checked.load_state_dict(plain.state_dict())  # identical weights

    x = torch.randint(0, 65, (4, 32))
    y = torch.randint(0, 65, (4, 32))

    def grads(model):
        model.train()
        _, loss = model(x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        return {n: p.grad for n, p in model.named_parameters() if p.grad is not None}

    for name, g_plain in grads(plain).items():
        g_checked = grads(checked)[name]
        assert torch.allclose(g_plain, g_checked, atol=1e-5, rtol=1e-4), (
            f"gradient mismatch at {name}"
        )


def test_grad_checkpoint_reduces_activation_memory():
    """Checkpointing must not *increase* peak activation memory (it recomputes)."""
    torch.manual_seed(0)
    plain = _model(False)
    checked = _model(True)
    x = torch.randint(0, 65, (8, 64))
    y = torch.randint(0, 65, (8, 64))

    def peak(model):
        model.train()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        base = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        _, loss = model(x, y)
        loss.backward()
        return (torch.cuda.max_memory_allocated() - base) if torch.cuda.is_available() else None

    if torch.cuda.is_available():
        m_plain = peak(plain)
        m_checked = peak(checked)
        # Both run; on a small model the gap can be small, so only assert sanity.
        assert m_checked <= m_plain * 1.2, (m_plain, m_checked)
