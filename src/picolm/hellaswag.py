"""Zero-shot HellaSwag evaluation (commonsense multiple choice).

HellaSwag is the standard zero-shot benchmark for language-model "common
sense": given a context sentence, pick the most plausible of four endings.
Chance accuracy is 25%.

The scoring method is the standard one (nanoGPT / GPT-2 papers): for each
ending, feed ``context + ending`` through the model and compute the average
negative log-likelihood of the *ending tokens given the context*. The ending
with the lowest NLL (highest likelihood) wins.

NOTE: this is a hard benchmark. A model trained on a tiny domain-specific
corpus (e.g. 1 MB of Shakespeare with a 65-character vocabulary) scores at
chance, because commonsense reasoning requires general-English pretraining at
scale. The harness is correct and ready for a larger model — the score simply
reports what the model can actually do.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def safe_encode(tokenizer, text: str) -> list[int]:
    """Encode text, tolerating characters outside a char tokenizer's vocab.

    A ``CharTokenizer`` raises ``KeyError`` on unseen characters; here they are
    dropped so an out-of-domain corpus (HellaSwag is modern English) can still
    be scored instead of crashing. A byte-level BPE tokenizer never fails.
    """
    try:
        return tokenizer.encode(text)
    except KeyError:
        if hasattr(tokenizer, "stoi"):
            return [tokenizer.stoi[c] for c in text if c in tokenizer.stoi]
        raise


@torch.no_grad()
def score_completion(
    model,
    ctx_tokens: list[int],
    end_tokens: list[int],
    block_size: int,
    device: torch.device,
) -> float:
    """Average negative log-likelihood of ``end_tokens`` given ``ctx_tokens``.

    Lower is better. The context is kept (left-truncated to ``block_size``) so
    the ending is always conditioned on as much context as fits.
    """
    full = (ctx_tokens + end_tokens)[-block_size:]
    n_end = min(len(end_tokens), len(full) - 1)

    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    logits, _ = model(x)  # (1, T-1, vocab)
    log_probs = F.log_softmax(logits, dim=-1)[0]  # (T-1, vocab)
    targets = torch.tensor(full[1:], dtype=torch.long, device=device)
    nll = -log_probs[torch.arange(len(targets)), targets]
    return float(nll[-n_end:].mean())


@torch.no_grad()
def evaluate_hellaswag(
    model,
    tokenizer,
    examples: Iterable[dict],
    device: torch.device,
    *,
    limit: int | None = None,
) -> dict:
    """Score a list of HellaSwag examples and report zero-shot accuracy.

    Each example is a dict with keys ``ctx`` (str), ``endings`` (list of 4
    strings), and ``label`` (int 0-3, the correct ending).
    """
    model.eval()
    block_size = model.config.block_size
    n_correct = 0
    n_total = 0
    for ex in examples:
        if limit is not None and n_total >= limit:
            break
        ctx_tokens = safe_encode(tokenizer, ex["ctx"])
        scores = [
            score_completion(
                model, ctx_tokens, safe_encode(tokenizer, e), block_size, device
            )
            for e in ex["endings"]
        ]
        pred = scores.index(min(scores))
        n_correct += int(pred == ex["label"])
        n_total += 1

    return {
        "accuracy": round(n_correct / n_total, 4) if n_total else 0.0,
        "correct": n_correct,
        "n": n_total,
        "chance": 0.25,
    }
