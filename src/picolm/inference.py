"""Inference engine: sampling, KV-cache decoding, and int8 quantization.

Three pieces that turn a *trained* model into something you can actually serve:

* :func:`sample` — temperature / top-k / top-p (nucleus) sampling.
* :func:`generate_kv` — autoregressive decoding with a key/value cache that
  avoids recomputing attention over the entire sequence each step. It hand-
  unrolls the transformer from the raw weights to make the mechanism explicit.
* :func:`quantize_int8` / :func:`dequantize_int8` — symmetric per-tensor int8
  weight quantization that shrinks the model ~4x with a single scale per
  weight matrix.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from picolm.model import GPT


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Sample a token index from logits with temperature, top-k, top-p."""
    logits = logits / max(temperature, 1e-6)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, [-1]]] = float("-inf")

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        to_remove = cumprobs > top_p
        to_remove[..., 1:] = to_remove[..., :-1].clone()
        to_remove[..., 0] = 0
        keep = torch.zeros_like(logits, dtype=torch.bool)
        keep.scatter_(1, sorted_idx, ~to_remove)
        logits[~keep] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# KV-cache decoding
# ---------------------------------------------------------------------------
def _layer_norm(x: torch.Tensor, ln) -> torch.Tensor:
    return F.layer_norm(x, ln.normalized_shape, ln.weight, ln.bias, ln.eps)


def _run_blocks(
    model: GPT, x: torch.Tensor, caches: list[tuple[torch.Tensor, torch.Tensor] | None]
) -> torch.Tensor:
    """Run the token embeddings ``x`` through every block, updating the cache.

    ``x`` has shape (B, T, C) where T is the number of *new* tokens this step.
    When the cache is warm, T == 1 and the new K/V are concatenated onto the
    cached K/V so attention sees the full history.
    """
    cfg = model.config
    head_size = cfg.n_embd // cfg.n_head

    for i, block in enumerate(model.transformer.h):
        # --- attention ---
        xn = _layer_norm(x, block.ln_1)
        qkv = F.linear(xn, block.attn.c_attn.weight, block.attn.c_attn.bias)
        q, k, v = qkv.split(cfg.n_embd, dim=2)
        B, T, _ = q.shape
        q = q.view(B, T, cfg.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, cfg.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, cfg.n_head, head_size).transpose(1, 2)

        past_length = 0
        if caches[i] is not None:
            pk, pv = caches[i]
            past_length = pk.size(2)
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        caches[i] = (k, v)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
        if T > 1:
            query_positions = past_length + torch.arange(T, device=q.device)
            key_positions = torch.arange(k.size(2), device=q.device)
            causal = query_positions[:, None] >= key_positions[None, :]
            att = att.masked_fill(~causal[None, None], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, cfg.n_embd)
        y = F.linear(y, block.attn.c_proj.weight, block.attn.c_proj.bias)
        x = x + y

        # --- MLP ---
        xn = _layer_norm(x, block.ln_2)
        h = F.gelu(
            F.linear(xn, block.mlp.c_fc.weight, block.mlp.c_fc.bias), approximate="tanh"
        )
        h = F.linear(h, block.mlp.c_proj.weight, block.mlp.c_proj.bias)
        x = x + h

    return x


@torch.no_grad()
def generate_kv(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Autoregressive generation with a key/value cache.

    Returns ``idx`` extended by ``max_new_tokens`` sampled tokens. For the
    learned-position/LayerNorm architecture and while the entire prompt plus
    generation stays inside ``block_size``, seeded output matches
    :meth:`GPT.generate` exactly. Unsupported configurations and lengths fail
    explicitly instead of silently claiming parity.
    """
    cfg = model.config
    if cfg.rope or cfg.rmsnorm:
        raise NotImplementedError(
            "generate_kv currently supports learned positions with LayerNorm only"
        )
    if idx.ndim != 2 or idx.size(1) == 0:
        raise ValueError("idx must have shape (batch, tokens) with at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if idx.size(1) + max_new_tokens > cfg.block_size:
        raise ValueError(
            "KV-cache parity is defined only while prompt + generation fits "
            f"inside block_size={cfg.block_size}"
        )
    model.eval()
    wte = model.transformer.wte.weight
    wpe = model.transformer.wpe.weight
    caches: list = [None] * cfg.n_layer

    # Warm the cache with the full prompt in one pass.
    T = idx.size(1)
    pos = torch.arange(0, T, device=idx.device)
    x = wte[idx] + wpe[pos]
    x = _run_blocks(model, x, caches)
    x = _layer_norm(x, model.transformer.ln_f)
    logits = F.linear(x, model.lm_head.weight, model.lm_head.bias)

    next_pos = T
    for _ in range(max_new_tokens):
        idx_next = sample(logits[:, -1, :], temperature, top_k, top_p)
        idx = torch.cat([idx, idx_next], dim=1)

        p = next_pos
        x = wte[idx_next] + wpe[p : p + 1]
        next_pos += 1

        x = _run_blocks(model, x, caches)
        x = _layer_norm(x, model.transformer.ln_f)
        logits = F.linear(x, model.lm_head.weight, model.lm_head.bias)

    return idx


# ---------------------------------------------------------------------------
# int8 quantization
# ---------------------------------------------------------------------------
def quantize_int8(
    model: GPT,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Symmetric per-tensor int8 quantization of every Linear weight.

    Returns ``(q_weights, scales)``. Each weight matrix ``w`` is stored as
    ``round(w / scale)`` in int8, with ``scale = max(|w|) / 127``.
    """
    q: dict[str, torch.Tensor] = {}
    scales: dict[str, float] = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            w = param.detach().float()
            amax = w.abs().max().item()
            scale = amax / 127.0 if amax > 0 else 1e-8
            q[name] = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
            scales[name] = scale
    return q, scales


def dequantize_int8(
    model: GPT, q: dict[str, torch.Tensor], scales: dict[str, float]
) -> None:
    """Write the de-quantized float32 weights back into ``model`` in place."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in q:
                restored = q[name].to(device=param.device, dtype=param.dtype)
                param.copy_(restored * scales[name])


def model_size_bytes(
    model: GPT,
    q: dict[str, torch.Tensor] | None = None,
    scales: dict[str, float] | None = None,
) -> dict[str, int]:
    """Report fp32 storage vs mixed int8/fp32 parameter storage.

    Matrix weights selected by :func:`quantize_int8` use one byte per value
    plus one fp32 scale per tensor. Parameters not selected for quantization
    (for example normalization vectors) remain fp32 and must still be counted.
    """
    fp32 = sum(p.numel() * 4 for p in model.parameters())
    result = {"fp32_bytes": fp32}
    if q is not None:
        int8 = sum(t.numel() * 1 for t in q.values())
        int8 += sum(4 for _ in scales.values()) if scales else 0
        int8 += sum(
            parameter.numel() * 4
            for name, parameter in model.named_parameters()
            if name not in q
        )
        result["int8_bytes"] = int8
        result["compression_ratio"] = round(fp32 / int8, 2)
    return result
