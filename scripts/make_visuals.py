#!/usr/bin/env python3
"""Render real model outputs into README/dashboard images.

Usage::

    python scripts/make_visuals.py --ckpt out/ckpt.pt --out assets
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picolm.cli import _load_tokenizer
from picolm.inference import quantize_int8
from picolm.model import GPT

BLUE = "#3b82f6"
ORANGE = "#f59e0b"


def tok_label(tok, i: int) -> str:
    ch = tok.decode([i])
    if ch == "\n":
        return "\\n"
    if ch == " ":
        return "·"
    return ch if ch else f"<{i}>"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--out", default="assets")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT.load(args.ckpt).to(device).eval()
    tok = _load_tokenizer(Path(args.ckpt).parent)

    # ---- attention map ----------------------------------------------------
    prompt = "To be, or not to be: that is the question."
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        _, maps = model.forward_with_attention(idx)
    attn = maps[0][0, 3].cpu().numpy()  # layer 0, head 3
    labels = [tok_label(tok, i) for i in ids]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(attn, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(labels, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel("attended token")
    ax.set_ylabel("query token")
    ax.set_title("Causal self-attention — layer 0, head 3")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attention weight")
    fig.tight_layout()
    fig.savefig(out / "attention.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out / 'attention.png'}")

    # ---- quantization histogram ------------------------------------------
    q, scales = quantize_int8(model)
    name = "transformer.h.0.attn.c_attn.weight"
    w = dict(model.named_parameters())[name].detach().float().cpu().numpy()
    deq = q[name].cpu().numpy().astype("float32") * scales[name]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.hist(w.ravel(), bins=120, alpha=0.55, label="float32", color=BLUE)
    ax.hist(deq.ravel(), bins=120, alpha=0.85, label="int8 (dequantized)", color=ORANGE)
    ax.set_xlabel("weight value")
    ax.set_ylabel("count")
    ax.set_title("int8 quantization: continuous weights snap to discrete buckets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "quantization.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out / 'quantization.png'}")

    # ---- next-token distribution -----------------------------------------
    prompt = "The king"
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, _ = model(idx)
    probs = torch.softmax(logits[0, -1], dim=-1)
    top_probs, top_ids = torch.topk(probs, 15)
    labels = [tok_label(tok, int(i)) for i in top_ids.cpu()]

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.bar(labels, top_probs.cpu().numpy(), color=BLUE)
    ax.set_ylabel("probability")
    ax.set_xlabel("next token")
    ax.set_title('Next-token distribution after "The king"')
    plt.setp(ax.get_xticklabels(), fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "sampling.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out / 'sampling.png'}")


if __name__ == "__main__":
    main()
