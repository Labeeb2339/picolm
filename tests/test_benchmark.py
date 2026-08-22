"""Measurement-contract tests for PicoLM benchmarks."""

from unittest.mock import Mock

import torch

from picolm.benchmark import _synchronize_if_cuda, quantization_impact
from picolm.config import ModelConfig
from picolm.model import GPT


def test_cuda_synchronization_is_explicit(monkeypatch):
    synchronize = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)

    _synchronize_if_cuda(torch.device("cuda"))
    _synchronize_if_cuda(torch.device("cpu"))

    synchronize.assert_called_once_with(torch.device("cuda"))


def test_quantization_compares_the_same_seeded_windows(monkeypatch):
    calls: list[int] = []

    def fake_perplexity(_model, _data, _block, _batch, _device, _count, seed):
        calls.append(seed)
        return 1.0, 4.0 + len(calls) / 100

    monkeypatch.setattr("picolm.benchmark.model_perplexity", fake_perplexity)
    model = GPT(ModelConfig(vocab_size=8, block_size=8, n_layer=1, n_head=1, n_embd=8))
    original = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    quantization_impact(
        model,
        torch.arange(100) % 8,
        block_size=8,
        batch_size=4,
        device=torch.device("cpu"),
        num_batches=2,
        seed=99,
    )

    assert calls == [99, 99]
    assert all(
        torch.equal(parameter, original[name])
        for name, parameter in model.named_parameters()
    )
