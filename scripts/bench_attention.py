"""Benchmark PicoLM's eager attention vs the Triton FlashAttention drop-in.

Run from the repo root:  python scripts/bench_attention.py
"""

from __future__ import annotations

import math

import torch
import triton.testing

from picolm.flash_attn import flash_attention


def eager_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """PicoLM's current attention (materializes the T x T score matrix)."""
    _B, _H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    s = (q @ k.transpose(-2, -1)) * scale
    causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    s = s.masked_fill(~causal, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ v


def main() -> None:
    device = "cuda"
    H, D = 6, 64  # PicoLM's exact head geometry (n_head=6, head_size=64)
    print(
        f"device={torch.cuda.get_device_name(0)} | PicoLM attention: H={H}, D={D}, fp16"
    )
    print(f"{'T':>5} {'eager':>9} {'flash':>9} {'speedup':>8}")
    print("-" * 34)
    for T in (64, 128, 256, 512, 1024):
        q = torch.randn(1, H, T, D, device=device, dtype=torch.float16)
        k = torch.randn(1, H, T, D, device=device, dtype=torch.float16)
        v = torch.randn(1, H, T, D, device=device, dtype=torch.float16)

        ms_eager = triton.testing.do_bench(
            lambda q=q, k=k, v=v: eager_attention(q, k, v)
        )
        ms_flash = triton.testing.do_bench(
            lambda q=q, k=k, v=v: flash_attention(q, k, v)
        )

        print(
            f"{T:>5} {ms_eager:>8.3f}m {ms_flash:>8.3f}m {ms_eager / ms_flash:>7.2f}x"
        )


if __name__ == "__main__":
    main()
