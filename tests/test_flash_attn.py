"""Correctness tests for the Triton FlashAttention drop-in (GPU-gated)."""

import math

import pytest
import torch

from picolm.flash_attn import flash_attention, flash_available

pytestmark = pytest.mark.skipif(not flash_available(), reason="requires Triton + CUDA")


def eager_attention(q, k, v):
    _B, _H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    s = (q @ k.transpose(-2, -1)) * scale
    causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    s = s.masked_fill(~causal, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ v


@pytest.mark.parametrize("T", [64, 128, 256])
def test_flash_matches_eager(T):
    torch.manual_seed(0)
    q = torch.randn(1, 6, T, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 6, T, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 6, T, 64, device="cuda", dtype=torch.float16)
    out_flash = flash_attention(q, k, v)
    out_eager = eager_attention(q, k, v)
    assert torch.allclose(out_flash.float(), out_eager.float(), atol=5e-2, rtol=5e-2), (
        f"max abs err {(out_flash.float() - out_eager.float()).abs().max().item():.3e}"
    )


def test_flash_fp32_cast_roundtrip():
    """fp32 input is cast to fp16 internally and the output is cast back."""
    torch.manual_seed(0)
    q = torch.randn(1, 6, 64, 64, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 6, 64, 64, device="cuda", dtype=torch.float32)
    v = torch.randn(1, 6, 64, 64, device="cuda", dtype=torch.float32)
    out = flash_attention(q, k, v)
    assert out.dtype == torch.float32
    assert out.shape == q.shape
    # matches fp16 eager within fp16 tolerance
    ref = eager_attention(q.half(), k.half(), v.half()).float()
    assert torch.allclose(out, ref, atol=5e-2, rtol=5e-2)
