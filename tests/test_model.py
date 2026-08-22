"""Model tests: forward shapes, causality, generation, parameter count."""

import torch

from picolm.config import ModelConfig
from picolm.model import GPT


def make_model() -> GPT:
    return GPT(
        ModelConfig(
            vocab_size=65,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
    )


def test_forward_shapes():
    model = make_model()
    idx = torch.randint(0, 65, (2, 16))
    logits, loss = model(idx)
    assert logits.shape == (2, 16, 65)
    assert loss is None

    logits, loss = model(idx, idx)
    assert loss is not None
    assert loss.ndim == 0  # scalar


def test_full_context_length():
    model = make_model()
    idx = torch.randint(0, 65, (1, 32))  # block_size == 32
    logits, _ = model(idx)
    assert logits.shape == (1, 32, 65)


def test_generate_shape():
    model = make_model().eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, 10, temperature=1.0)
    assert out.shape == (1, 11)


def test_param_count_positive():
    model = make_model()
    assert model.get_num_params() > 0
    assert model.get_num_params(non_embedding=True) > 0


def test_save_load_roundtrip(tmp_path):
    model = make_model()
    path = tmp_path / "ckpt.pt"
    model.save(str(path))
    loaded = GPT.load(str(path))
    assert loaded.config.n_layer == 2
    # Same weights -> same logits.
    idx = torch.randint(0, 65, (1, 8))
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = loaded(idx)
    assert torch.allclose(a, b, atol=1e-6)


def test_checkpoint_loader_defaults_to_cpu(tmp_path):
    model = make_model()
    path = tmp_path / "ckpt.pt"
    model.save(str(path))

    loaded = GPT.load(str(path))

    assert loaded.device.type == "cpu"


def test_causal_masking():
    """Perturbing future tokens must not change past-token logits."""
    model = make_model().eval()
    idx = torch.randint(0, 65, (1, 16))
    with torch.no_grad():
        logits_a, _ = model(idx)

    # Corrupt the last two positions (future relative to positions 0..13).
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % 65
    idx2[0, -2] = (idx2[0, -2] + 2) % 65
    with torch.no_grad():
        logits_b, _ = model(idx2)

    # Positions 0..13 must be identical; only 14..15 may change.
    assert torch.allclose(logits_a[0, :14], logits_b[0, :14], atol=1e-6)
    assert not torch.allclose(logits_a[0, -1], logits_b[0, -1])


def test_forward_with_attention():
    """Attention maps are returned with the right shape and causal mask."""
    model = make_model().eval()
    idx = torch.randint(0, 65, (1, 8))
    logits, maps = model.forward_with_attention(idx)
    assert logits.shape == (1, 8, 65)
    assert len(maps) == 2  # n_layer == 2
    for m in maps:
        assert m.shape == (1, 2, 8, 8)  # (B, n_head, T, T)
        # Strictly upper triangle (future positions) must be exactly zero.
        assert m[0, 0].triu(diagonal=1).abs().max() == 0.0
