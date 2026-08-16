"""A GPT-style decoder-only transformer, implemented from scratch.

The architecture follows GPT-2 (Radford et al., 2019): token + positional
embeddings, a stack of transformer blocks (pre-norm LayerNorm, multi-head
causal self-attention, GELU MLP), final LayerNorm, and a weight-tied language
modelling head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from picolm.config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (LLaMA-style) — normalize by RMS, no bias.

    Cheaper than LayerNorm (no mean subtraction) and matches the modern
    transformer recipe (LLaMA/Mistral/Gemma). ``x -> x / rms(x) * weight``.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def precompute_rope(head_size: int, max_len: int, base: float = 10000.0):
    """Precompute rotary-embedding cos/sin tables (shape ``(max_len, head_size//2)``)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))
    pos = torch.arange(max_len).float()
    angles = pos[:, None] * inv_freq[None, :]
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to ``x`` of shape ``(B, H, T, D)``.

    Rotates each pair ``(i, i + D/2)`` by ``angle[i]`` (the half-split / GPT-NeoX
    RoPE convention). This is a norm-preserving rotation that encodes *relative*
    position in the attention dot product, so it generalizes to sequences longer
    than the training window better than learned absolute embeddings.
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    cos = cos[None, None]  # (1, 1, T, D//2)
    sin = sin[None, None]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (lower-triangular) mask.

    The mask is stored as a buffer so the forward pass never allocates it.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        # q, k, v in a single projection for efficiency.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        )
        self.register_buffer("mask", mask)

        # Optional attention-weight capture for visualization.
        self.record_attn = False
        self.attn_weights: torch.Tensor | None = None

        # Optional Triton FlashAttention path (off by default). Requires Triton
        # + CUDA, and it casts to fp16 — so it's for fp16/bf16 inference where
        # T is divisible by 64. Falls back to eager attention otherwise.
        self.use_flash = False

        # Optional rotary position embeddings (RoPE, LLaMA-style).
        self.use_rope = config.rope
        if self.use_rope:
            cos, sin = precompute_rope(self.head_size, config.block_size)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
            k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])

        # Optional Triton FlashAttention path (single-pass, O(N) memory).
        if self.use_flash and T % 64 == 0:
            from picolm.flash_attn import flash_attention, flash_available

            if flash_available():
                y = flash_attention(q, k, v)  # (B, n_head, T, head_size)
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                y = self.resid_dropout(self.c_proj(y))
                return y

        # Scaled dot-product attention (eager fallback).
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        if self.record_attn:
            self.attn_weights = att.detach()
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_head, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """A single transformer block: pre-norm attention + pre-norm MLP."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.rmsnorm:
            self.ln_1 = RMSNorm(config.n_embd)
            self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Decoder-only language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd) if config.rmsnorm else nn.LayerNorm(config.n_embd, bias=config.bias),
        )
        if not config.rope:
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the output projection shares the input embedding matrix.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scale residual projections down (GPT-2 / nanoGPT recipe).
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and "wpe" in self.transformer:
            n -= self.transformer.wpe.weight.numel()  # positional embeddings
        return n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            idx: token ids of shape (B, T).
            targets: optional labels of shape (B, T) for loss computation.

        Returns:
            (logits, loss) where loss is None when targets is None.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )

        tok_emb = self.transformer.wte(idx)  # (B, T, C)
        if self.config.rope:
            x = self.transformer.drop(tok_emb)  # RoPE injects position in attention
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.transformer.wpe(pos)  # (T, C)
            x = self.transformer.drop(tok_emb + pos_emb)

        if self.config.grad_checkpoint and self.training:
            for block in self.transformer.h:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.transformer.h:
                x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def forward_with_attention(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass that also returns per-layer attention maps.

        Returns ``(logits, attention_maps)`` where ``attention_maps`` is a list
        with one ``(B, n_head, T, T)`` tensor per transformer block.
        """
        blocks = self.transformer.h
        for block in blocks:
            block.attn.record_attn = True
        logits, _ = self.forward(idx)
        maps = [block.attn.attn_weights for block in blocks]
        for block in blocks:
            block.attn.record_attn = False
        return logits, maps

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation (temperature + optional top-k / top-p)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = 0
                keep = torch.zeros_like(logits, dtype=torch.bool)
                keep.scatter_(1, sorted_idx, ~remove)
                logits[~keep] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: tuple[float, float]
    ) -> torch.optim.Optimizer:
        """AdamW with weight decay applied only to 2D+ (matrix) parameters."""
        decay = [p for p in self.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {"config": self.config, "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str | None = None) -> "GPT":
        # weights_only=False because the checkpoint bundles the ModelConfig
        # dataclass alongside the state_dict (a trusted, self-produced file).
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
