"""FlashAttention (causal) for PicoLM, in Triton.

A fused single-pass attention kernel that never materializes the ``T x T``
score matrix (O(N^2) memory -> O(N)). This is an optional drop-in for
:class:`CausalSelfAttention`; it falls back to eager attention when Triton is
unavailable or the dtype/sequence length is unsupported.

The kernel is the same structure as ``pico-kernels/attention.py``: tiled
``Q K^T`` with an online-softmax (running max + sum) and a fused ``P V``.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - no GPU / no triton
    _HAS_TRITON = False


def flash_available() -> bool:
    """True when the Triton FlashAttention path can be used on this box."""
    return _HAS_TRITON and torch.cuda.is_available()


if _HAS_TRITON:

    @triton.jit
    def _attn_fwd(
        Q, K, V, O, sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vk, stride_vn,
        stride_ob, stride_oh, stride_om, stride_on,
        B, H, N_CTX,
        HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_m_tiles = tl.cdiv(N_CTX, BLOCK_M)
        m_tile = pid % num_m_tiles
        tmp = pid // num_m_tiles
        h = tmp % H
        b = tmp // H

        offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)

        q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(q_ptrs)

        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

        hi = (m_tile + 1) * BLOCK_M
        for start_n in range(0, hi, BLOCK_N):
            k_ptrs = K + b * stride_kb + h * stride_kh + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k = tl.load(k_ptrs)

            qk = tl.dot(q, tl.trans(k)) * sm_scale
            mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
            qk = tl.where(mask, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v_ptrs = V + b * stride_vb + h * stride_vh + (start_n + offs_n)[:, None] * stride_vk + offs_d[None, :] * stride_vn
            v = tl.load(v_ptrs)
            acc = tl.dot(p.to(v.dtype), v, acc)
            m_i = m_ij

        acc = acc / l_i[:, None]
        o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
        tl.store(o_ptrs, acc.to(O.dtype.element_ty))


def flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    BLOCK_M: int = 64, BLOCK_N: int = 64,
) -> torch.Tensor:
    """Causal FlashAttention on ``(B, H, T, D)`` tensors.

    Requires ``T`` divisible by the block sizes. fp32 inputs are cast to fp16
    for the tensor cores and the result is cast back, so this is a true drop-in
    for PicoLM's fp32 model (at fp16 attention precision, the same tradeoff a
    real fp16/bf16 inference stack makes).
    """
    if not flash_available():
        raise RuntimeError("Triton FlashAttention is not available on this box")

    assert q.dim() == 4 and q.shape == k.shape == v.shape
    B, H, T, D = q.shape
    assert T % BLOCK_M == 0 and T % BLOCK_N == 0, "T must be divisible by the block sizes"

    orig_dtype = q.dtype
    if orig_dtype not in (torch.float16, torch.bfloat16):
        q, k, v = q.half(), k.half(), v.half()

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)

    grid = (B * H * triton.cdiv(T, BLOCK_M),)
    _attn_fwd[grid](
        q, k, v, o, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, T,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return o.to(orig_dtype)
