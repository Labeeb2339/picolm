"""Benchmarks: decode speed and quantization error.

Two claims a reviewer will want checked with numbers:

* the KV-cache decoder is actually faster than recomputing attention (and by
  how much, for a given generation length);
* int8 quantization shrinks the model ~4x while barely moving perplexity.
"""

from __future__ import annotations

import time

import torch

from picolm.eval import model_perplexity
from picolm.inference import dequantize_int8, generate_kv, quantize_int8


def _synchronize_if_cuda(device: torch.device) -> None:
    """Finish queued CUDA work so wall-clock timings are meaningful."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_generation(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 40,
    warmup: int = 20,
) -> dict:
    """Time eager (recompute) vs KV-cache decoding in tokens/sec."""
    model.eval()

    # Warm up both paths before timing so compilation/cache effects are excluded.
    with torch.inference_mode():
        model.generate(idx, warmup, temperature, top_k)
        generate_kv(model, idx, warmup, temperature, top_k)
    _synchronize_if_cuda(idx.device)

    with torch.inference_mode():
        _synchronize_if_cuda(idx.device)
        t0 = time.perf_counter()
        model.generate(idx, max_new_tokens, temperature, top_k)
        _synchronize_if_cuda(idx.device)
        eager_s = time.perf_counter() - t0

        _synchronize_if_cuda(idx.device)
        t0 = time.perf_counter()
        generate_kv(model, idx, max_new_tokens, temperature, top_k)
        _synchronize_if_cuda(idx.device)
        kv_s = time.perf_counter() - t0

    return {
        "max_new_tokens": max_new_tokens,
        "prompt_tokens": int(idx.size(1)),
        "eager_seconds": round(eager_s, 3),
        "kv_seconds": round(kv_s, 3),
        "eager_tokens_per_sec": round(max_new_tokens / eager_s, 1),
        "kv_tokens_per_sec": round(max_new_tokens / kv_s, 1),
        "speedup": round(eager_s / kv_s, 2),
    }


def quantization_impact(
    model,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    num_batches: int = 50,
    seed: int = 42,
) -> dict:
    """Compare fp32 and int8 on the same seeded validation windows."""
    from picolm.inference import model_size_bytes

    _, ppl_fp32 = model_perplexity(
        model, data, block_size, batch_size, device, num_batches, seed
    )

    original = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if "weight" in name and parameter.dim() >= 2
    }
    q, scales = quantize_int8(model)
    size = model_size_bytes(model, q, scales)
    try:
        dequantize_int8(model, q, scales)
        _, ppl_int8 = model_perplexity(
            model, data, block_size, batch_size, device, num_batches, seed
        )
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in original:
                    parameter.copy_(original[name])

    delta = ppl_int8 - ppl_fp32
    return {
        "perplexity_fp32": round(ppl_fp32, 3),
        "perplexity_int8": round(ppl_int8, 3),
        "perplexity_delta": round(delta, 3),
        "perplexity_delta_percent": round(100.0 * delta / ppl_fp32, 3),
        "fp32_mb": round(size["fp32_bytes"] / 1e6, 2),
        "int8_mb": round(size["int8_bytes"] / 1e6, 2),
        "compression_ratio": size["compression_ratio"],
    }
