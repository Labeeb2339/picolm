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

    # Warm up CUDA kernels before timing.
    with torch.inference_mode():
        model.generate(idx, warmup, temperature, top_k)

    with torch.inference_mode():
        t0 = time.perf_counter()
        model.generate(idx, max_new_tokens, temperature, top_k)
        eager_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        generate_kv(model, idx, max_new_tokens, temperature, top_k)
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
) -> dict:
    """Measure perplexity before and after a float32 -> int8 -> float32 trip."""
    from picolm.inference import model_size_bytes

    _, ppl_fp32 = model_perplexity(
        model, data, block_size, batch_size, device, num_batches
    )

    q, scales = quantize_int8(model)
    size = model_size_bytes(model, q, scales)
    dequantize_int8(model, q, scales)

    _, ppl_int8 = model_perplexity(
        model, data, block_size, batch_size, device, num_batches
    )

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
