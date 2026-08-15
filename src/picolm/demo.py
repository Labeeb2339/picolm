"""PicoLM visualization dashboard.

Launch with ``picolm demo`` (or ``streamlit run src/picolm/demo.py``).

Five views that make the internals of a transformer tangible:

* **Generate** — interactive text generation.
* **Attention** — per-layer, per-head attention maps (the causal triangle).
* **Sampling** — the next-token probability distribution under
  temperature / top-k / top-p.
* **Quantization** — fp32 weights snapping to int8 buckets, plus the size and
  perplexity impact.
* **Training** — the loss curve and the cosine learning-rate schedule.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from picolm import _version
from picolm.cli import _load_tokenizer
from picolm.inference import model_size_bytes, quantize_int8
from picolm.model import GPT

st.set_page_config(page_title="PicoLM", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------------------
# Loading (cached so the checkpoint is read once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load(ckpt_path: str):
    model = GPT.load(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    tok = _load_tokenizer(Path(ckpt_path).parent)
    return model, tok, device


@st.cache_resource
def load_metrics(ckpt_path: str) -> dict | None:
    p = Path(ckpt_path).parent / "metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def tok_label(tok, i: int) -> str:
    """A short, display-safe label for a token id."""
    ch = tok.decode([i])
    if ch == "\n":
        return "\\n"
    if ch == " ":
        return "·"
    return ch if ch else f"<{i}>"


# ---------------------------------------------------------------------------
# Header + sidebar
# ---------------------------------------------------------------------------
st.title("🧠 PicoLM — internals, visualized")
st.caption(
    f"GPT-style LM built from scratch · v{_version.__version__} · "
    "attention, sampling, and quantization made visible"
)

with st.sidebar:
    st.header("Model")
    ckpt = st.text_input("Checkpoint", value="out/ckpt.pt")
    if not Path(ckpt).exists():
        st.warning(f"No checkpoint at `{ckpt}`. Train one first:\n\n"
                   "`picolm train --out-dir out`")
        st.stop()
    model, tok, device = load(ckpt)
    metrics = load_metrics(ckpt)
    st.success(f"Loaded · device={device} · {model.get_num_params():,} params")
    if metrics:
        st.caption(
            f"val loss {metrics['best_val_loss']:.3f} · "
            f"{metrics['max_iters']:,} iters · {metrics['dtype']}"
        )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_gen, tab_attn, tab_samp, tab_quant, tab_train = st.tabs(
    ["Generate", "Attention", "Sampling", "Quantization", "Training"]
)

# ===========================================================================
# Generate
# ===========================================================================
with tab_gen:
    st.subheader("Text generation")
    c1, c2 = st.columns([3, 1])
    with c1:
        prompt = st.text_area("Prompt", value="To be, or not to be", height=100)
    with c2:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.8, 0.05)
        top_k = st.slider("Top-k", 0, 200, 40, 5)
        top_p = st.slider("Top-p (0 = off)", 0.0, 1.0, 0.0, 0.05)
        max_tokens = st.slider("Max tokens", 10, 500, 200, 10)
        seed = st.number_input("Seed (0 = random)", 0, 2**31 - 1, 0)

    if st.button("Generate", type="primary"):
        if seed:
            torch.manual_seed(seed)
        ids = tok.encode(prompt) if prompt else [0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model.generate(
                idx, max_tokens, temperature,
                top_k if top_k > 0 else None,
            )
        st.text_area("Output", value=tok.decode(out[0].tolist()), height=220)

# ===========================================================================
# Attention
# ===========================================================================
with tab_attn:
    st.subheader("Attention maps")
    st.caption(
        "Each row is a token, each column is what it attends to. The "
        "lower-triangular shape is the causal mask: a token can only look at "
        "itself and the past."
    )
    prompt = st.text_area("Prompt", value="To be, or not to be", key="attn_prompt",
                          height=80)

    n_layer = model.config.n_layer
    n_head = model.config.n_head
    layer = st.slider("Layer", 0, n_layer - 1, 0)
    head = st.slider("Head", 0, n_head - 1, 0)

    ids = tok.encode(prompt) if prompt else [0]
    T = len(ids)
    if T > model.config.block_size:
        ids = ids[-model.config.block_size:]
        T = len(ids)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    _, maps = model.forward_with_attention(idx)
    attn = maps[layer][0, head].cpu().numpy()  # (T, T)

    labels = [tok_label(tok, i) for i in ids]
    fig, ax = plt.subplots(figsize=(max(6, T * 0.4), max(5, T * 0.4)))
    im = ax.imshow(attn, cmap="viridis", aspect="auto")
    ax.set_xticks(range(T))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(T))
    ax.set_yticklabels(labels, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel("attended token")
    ax.set_ylabel("query token")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attention weight")
    st.pyplot(fig)

# ===========================================================================
# Sampling
# ===========================================================================
with tab_samp:
    st.subheader("Next-token probability distribution")
    st.caption(
        "The model's raw confidence over the vocabulary for the very next "
        "token. Watch temperature flatten the distribution and top-k/top-p "
        "prune the long tail."
    )
    prompt = st.text_area("Prompt", value="The king", key="samp_prompt", height=70)
    c1, c2, c3 = st.columns(3)
    t = c1.slider("Temperature", 0.0, 2.0, 1.0, 0.05, key="samp_t")
    k = c2.slider("Top-k", 0, 200, 0, 5, key="samp_k")
    p = c3.slider("Top-p (0 = off)", 0.0, 1.0, 0.0, 0.05, key="samp_p")

    ids = tok.encode(prompt) if prompt else [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, _ = model(idx)
    last = logits[0, -1].float()

    # Apply the sampling transforms exactly as inference.sample does.
    scaled = last / max(t, 1e-6)
    if k > 0:
        kk = min(k, scaled.size(-1))
        v, _ = torch.topk(scaled, kk)
        scaled[scaled < v[-1]] = float("-inf")
    if 0.0 < p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        keep_mask = torch.zeros_like(scaled, dtype=torch.bool)
        remove = cum > p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        keep_mask.scatter_(0, sorted_idx, ~remove)
        scaled[~keep_mask] = float("-inf")

    probs = torch.softmax(scaled, dim=-1)
    top_probs, top_ids = torch.topk(probs, 20)
    labels = [tok_label(tok, int(i)) for i in top_ids.cpu()]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(labels, top_probs.cpu().numpy(), color="tab:blue")
    ax.set_ylabel("probability")
    ax.set_xlabel("token")
    ax.set_title("Top-20 next-token probabilities (after sampling transforms)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)

    entropy = float(-(probs * torch.log(probs + 1e-12)).sum())
    st.metric("Distribution entropy (bits)", f"{entropy:.3f}")

# ===========================================================================
# Quantization
# ===========================================================================
with tab_quant:
    st.subheader("int8 weight quantization")
    st.caption(
        "Each weight matrix is scaled by `scale = max|w| / 127` and rounded to "
        "int8. Watch the smooth float32 histogram snap onto 255 discrete "
        "buckets — that's the whole trick."
    )

    q, scales = quantize_int8(model)
    size = model_size_bytes(model, q, scales)

    names = [n for n, p in model.named_parameters()
             if "weight" in n and p.dim() >= 2]
    sel = st.selectbox("Weight matrix", names)
    w = dict(model.named_parameters())[sel].detach().float().cpu().numpy()
    deq = q[sel].cpu().numpy().astype(np.float32) * scales[sel]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(w.ravel(), bins=120, alpha=0.55, label="float32", color="tab:blue")
    ax.hist(deq.ravel(), bins=120, alpha=0.85, label="int8 (dequantized)",
            color="tab:orange")
    ax.set_xlabel("weight value")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Scale", f"{scales[sel]:.4g}")
    c2.metric("Unique int8 values", f"{len(np.unique(q[sel].cpu().numpy()))}")
    c3.metric("Max |weight|", f"{np.abs(w).max():.4g}")

    st.divider()
    st.markdown("**Whole-model footprint**")
    c1, c2, c3 = st.columns(3)
    c1.metric("float32", f"{size['fp32_bytes'] / 1e6:.2f} MB")
    c2.metric("int8", f"{size['int8_bytes'] / 1e6:.2f} MB")
    c3.metric("Compression", f"{size['compression_ratio']:.2f}×")

# ===========================================================================
# Training
# ===========================================================================
with tab_train:
    st.subheader("Training dynamics")
    if not metrics:
        st.info("No `metrics.json` found next to the checkpoint.")
    else:
        st.markdown("**Loss curve** — train and validation loss over training.")
        import pandas as pd

        eval_interval = metrics.get("eval_interval", 250)
        xs = [i * eval_interval for i in range(len(metrics["val_loss"]))]
        st.line_chart(pd.DataFrame({
            "train": metrics["train_loss"],
            "val": metrics["val_loss"],
        }, index=xs))

        st.markdown(
            "**Learning-rate schedule** — cosine decay with warmup "
            "(the exact curve used to train this model)."
        )
        warmup = 100
        lr_max = metrics.get("learning_rate", 3e-4)
        lr_min = lr_max * 0.1
        iters = metrics.get("max_iters", 2000)
        xs2 = list(range(0, iters + 1))
        lrs = []
        for it in xs2:
            if it < warmup:
                lrs.append(lr_max * (it + 1) / warmup)
            elif it > iters:
                lrs.append(lr_min)
            else:
                prog = (it - warmup) / max(1, iters - warmup)
                lrs.append(lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * prog)))
        st.line_chart(pd.DataFrame({"lr": lrs}, index=xs2))

st.markdown("---")
st.caption("Trained on tiny Shakespeare. Educational from-scratch GPT — see "
           "MODEL_CARD.md for methodology and limitations.")
