"""Evaluation: perplexity and baseline comparison.

A loss number only means something next to a baseline. This module reports:

* model perplexity on held-out data (perplexity = exp(cross-entropy loss));
* a unigram baseline (predict the most frequent character);
* a bigram baseline (predict the next character from the previous one).

If the transformer does not clearly beat the bigram table, something is wrong
with training or evaluation — that comparison is what makes the result
credible.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from picolm.training import get_batch


@torch.no_grad()
def model_perplexity(
    model,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    num_batches: int = 100,
) -> tuple[float, float]:
    """Return ``(avg_loss, perplexity)`` on ``data`` over ``num_batches``."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for _ in range(num_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


def unigram_baseline(
    train_data: torch.Tensor, val_data: torch.Tensor, vocab_size: int
) -> tuple[float, float]:
    """Perplexity of a character-frequency (unigram) model with Laplace smoothing."""
    counts = torch.bincount(train_data, minlength=vocab_size).float()
    probs = (counts + 1.0) / (counts.sum() + vocab_size)
    log_probs = torch.log(probs)
    nll = -log_probs[val_data].sum().item()
    avg_loss = nll / val_data.numel()
    return avg_loss, math.exp(avg_loss)


def bigram_baseline(
    train_data: torch.Tensor, val_data: torch.Tensor, vocab_size: int, alpha: float = 1.0
) -> tuple[float, float]:
    """Perplexity of a character-level bigram model with add-``alpha`` smoothing."""
    import numpy as np

    # Vectorized (prev, next) pair counting via numpy fancy-index accumulate.
    counts_np = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    np.add.at(counts_np, (train_data[:-1].numpy(), train_data[1:].numpy()), 1.0)
    counts = torch.from_numpy(counts_np).float()

    row_totals = counts.sum(dim=1, keepdim=True) + alpha * vocab_size
    log_probs = torch.log((counts + alpha) / row_totals)

    vprev = val_data[:-1].numpy()
    vcur = val_data[1:].numpy()
    nll = float(-log_probs[vprev, vcur].sum().item())
    avg_loss = nll / (len(val_data) - 1)
    return avg_loss, math.exp(avg_loss)


def evaluate(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    *,
    val_frac: float = 0.1,
    block_size: int | None = None,
    batch_size: int = 64,
    num_batches: int = 100,
) -> dict:
    """Full evaluation: model perplexity vs unigram and bigram baselines."""
    block_size = block_size or model.config.block_size
    vocab_size = model.config.vocab_size

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int((1.0 - val_frac) * len(ids))
    train_data = ids[:n]
    val_data = ids[n:]

    model_avg_loss, model_ppl = model_perplexity(
        model, val_data, block_size, batch_size, device, num_batches
    )
    uni_loss, uni_ppl = unigram_baseline(train_data, val_data, vocab_size)
    bi_loss, bi_ppl = bigram_baseline(train_data, val_data, vocab_size)

    return {
        "model_loss": round(model_avg_loss, 4),
        "model_perplexity": round(model_ppl, 2),
        "unigram_perplexity": round(uni_ppl, 2),
        "bigram_perplexity": round(bi_ppl, 2),
        "bigram_vs_model": round(bi_ppl / model_ppl, 2),
        "unigram_vs_model": round(uni_ppl / model_ppl, 2),
        "val_tokens": int(len(val_data)),
        "num_batches": num_batches,
    }
