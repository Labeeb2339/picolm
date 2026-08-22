"""Inference tests: sampling, KV-cache correctness, int8 quantization."""

import pytest
import torch

from picolm.config import ModelConfig
from picolm.inference import (
    dequantize_int8,
    generate_kv,
    model_size_bytes,
    quantize_int8,
    sample,
)
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
    ).eval()


def test_sample_argmax_at_low_temperature():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    out = sample(logits, temperature=0.01)
    assert out.shape == (1, 1)
    assert out.item() == 2


def test_sample_top_k_respects_k():
    torch.manual_seed(0)
    logits = torch.tensor([[0.1, 0.2, 10.0, 0.3]])
    # top_k=1 collapses to the argmax deterministically.
    out = sample(logits, temperature=1.0, top_k=1)
    assert out.item() == 2


def test_kv_cache_matches_model_generate():
    torch.manual_seed(123)
    model = make_model()
    idx = torch.randint(0, 65, (1, 5))

    torch.manual_seed(7)
    ref = model.generate(idx, 20, temperature=1.0)

    torch.manual_seed(7)
    kv = generate_kv(model, idx, 20, temperature=1.0)

    assert ref.shape == kv.shape == (1, 25)
    assert torch.equal(ref, kv)


def test_kv_cache_greedy_prefill_matches_eager():
    torch.manual_seed(321)
    model = make_model()
    idx = torch.randint(0, 65, (1, 12))

    eager = model.generate(idx.clone(), 8, temperature=1.0, top_k=1)
    cached = generate_kv(model, idx.clone(), 8, temperature=1.0, top_k=1)

    assert torch.equal(eager, cached)


def test_kv_cache_rejects_lengths_outside_its_parity_contract():
    model = make_model()
    idx = torch.zeros((1, 30), dtype=torch.long)
    with pytest.raises(ValueError, match="fits inside block_size"):
        generate_kv(model, idx, 3)


def test_kv_cache_rejects_unsupported_modern_configuration():
    model = GPT(
        ModelConfig(
            vocab_size=65,
            block_size=32,
            n_layer=1,
            n_head=2,
            n_embd=16,
            rmsnorm=True,
            rope=True,
        )
    ).eval()
    with pytest.raises(NotImplementedError, match="learned positions"):
        generate_kv(model, torch.zeros((1, 1), dtype=torch.long), 1)


def test_quantization_reduces_size_and_preserves_forward():
    torch.manual_seed(0)
    model = make_model()
    q, scales = quantize_int8(model)
    size = model_size_bytes(model, q, scales)
    assert size["int8_bytes"] < size["fp32_bytes"]
    assert size["compression_ratio"] > 3.0
    expected_int8 = sum(tensor.numel() for tensor in q.values()) + 4 * len(scales)
    expected_int8 += sum(
        parameter.numel() * 4
        for name, parameter in model.named_parameters()
        if name not in q
    )
    assert size["int8_bytes"] == expected_int8

    dequantize_int8(model, q, scales)
    idx = torch.randint(0, 65, (1, 8))
    with torch.no_grad():
        logits, _ = model(idx)
    assert logits.shape == (1, 8, 65)
    assert torch.isfinite(logits).all()
