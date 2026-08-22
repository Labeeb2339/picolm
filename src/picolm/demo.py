"""Presenter-friendly PicoLM visualization dashboard.

Launch with ``picolm demo`` (or ``streamlit run src/picolm/demo.py``).
The dashboard keeps each view lazy so changing one control does not execute
every model visualization in the application.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from picolm import _version
from picolm.cli import _load_tokenizer
from picolm.inference import model_size_bytes, quantize_int8
from picolm.model import GPT

st.set_page_config(
    page_title="PicoLM Systems Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

_PICO_CSS = """
<style>
    :root {
        --pico-bg: #070a12;
        --pico-panel: rgba(18, 24, 41, 0.78);
        --pico-line: rgba(148, 163, 184, 0.18);
        --pico-text: #f8fafc;
        --pico-muted: #9aa7bc;
        --pico-violet: #8b5cf6;
        --pico-cyan: #22d3ee;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 78% -10%, rgba(139, 92, 246, .22), transparent 34rem),
            radial-gradient(circle at 8% 8%, rgba(34, 211, 238, .13), transparent 28rem),
            var(--pico-bg);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1220 0%, #080c16 100%);
        border-right: 1px solid var(--pico-line);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
    .block-container { max-width: 1280px; padding-top: 1.5rem; padding-bottom: 3rem; }
    .pico-hero {
        position: relative;
        overflow: hidden;
        padding: 2.2rem 2.35rem;
        margin-bottom: 1.25rem;
        border: 1px solid rgba(139, 92, 246, .32);
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(139, 92, 246, .18), rgba(34, 211, 238, .06)),
            rgba(9, 14, 27, .86);
        box-shadow: 0 26px 70px rgba(0, 0, 0, .34);
    }
    .pico-hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -80px;
        top: -120px;
        border-radius: 50%;
        background: rgba(34, 211, 238, .12);
    }
    .pico-kicker {
        color: var(--pico-cyan);
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .18em;
        text-transform: uppercase;
        margin-bottom: .7rem;
    }
    .pico-hero h1 {
        color: var(--pico-text);
        font-size: clamp(2rem, 4vw, 3.65rem);
        line-height: 1.03;
        letter-spacing: -.045em;
        margin: 0 0 .8rem 0;
        max-width: 840px;
    }
    .pico-gradient {
        background: linear-gradient(90deg, #c4b5fd, #67e8f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .pico-hero p { color: #b7c2d5; max-width: 820px; font-size: 1.05rem; margin: 0; }
    .pico-chips { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.3rem; }
    .pico-chip {
        border: 1px solid var(--pico-line);
        background: rgba(15, 23, 42, .64);
        color: #dbeafe;
        border-radius: 999px;
        padding: .36rem .7rem;
        font-size: .78rem;
        font-weight: 650;
    }
    .pico-section {
        color: var(--pico-cyan);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .28rem;
    }
    .pico-card {
        height: 100%;
        padding: 1.1rem 1.15rem;
        border: 1px solid var(--pico-line);
        border-radius: 16px;
        background: var(--pico-panel);
    }
    .pico-card strong { color: var(--pico-text); font-size: 1rem; }
    .pico-card p { color: var(--pico-muted); margin: .4rem 0 0; font-size: .9rem; }
    .pico-flow {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: .65rem;
        margin: 1rem 0 1.5rem;
    }
    .pico-flow-step {
        padding: 1rem .7rem;
        text-align: center;
        border-radius: 14px;
        border: 1px solid var(--pico-line);
        background: rgba(15, 23, 42, .7);
        color: #dbeafe;
        font-weight: 700;
    }
    .pico-flow-step span { display: block; color: var(--pico-muted); font-size: .72rem; margin-top: .2rem; }
    [data-testid="stMetric"] {
        padding: .85rem 1rem;
        border-radius: 15px;
        border: 1px solid var(--pico-line);
        background: rgba(15, 23, 42, .72);
    }
    [data-testid="stMetricValue"] { color: #f8fafc; }
    div[role="radiogroup"] {
        gap: .35rem;
        padding: .35rem;
        margin: .15rem 0 1.2rem;
        border: 1px solid var(--pico-line);
        border-radius: 14px;
        background: rgba(9, 14, 27, .72);
    }
    div[role="radiogroup"] label { padding: .42rem .65rem; border-radius: 10px; }
    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(139, 92, 246, .55);
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white;
        font-weight: 750;
        box-shadow: 0 8px 24px rgba(99, 102, 241, .22);
    }
    div[data-testid="stAlert"] { border-radius: 14px; }
    @media (max-width: 760px) {
        .pico-hero { padding: 1.55rem; }
        .pico-flow { grid-template-columns: 1fr; }
    }
</style>
"""
st.markdown(_PICO_CSS, unsafe_allow_html=True)


@st.cache_resource
def load(ckpt_path: str):
    """Load the model and tokenizer once per checkpoint."""

    model = GPT.load(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    tokenizer = _load_tokenizer(Path(ckpt_path).parent)
    return model, tokenizer, device


@st.cache_resource
def load_quantization(ckpt_path: str):
    """Build the deterministic int8 storage view once per checkpoint."""

    cpu_model = GPT.load(ckpt_path).cpu().eval()
    q_weights, scales = quantize_int8(cpu_model)
    return q_weights, scales, model_size_bytes(cpu_model, q_weights, scales)


@st.cache_data
def load_metrics(ckpt_path: str) -> dict | None:
    path = Path(ckpt_path).parent / "metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def tok_label(tokenizer, token_id: int) -> str:
    """Return a short, display-safe label for a token id."""

    token = tokenizer.decode([token_id])
    if token == "\n":
        return "\\n"
    if token == " ":
        return "·"
    return token if token else f"<{token_id}>"


def encode_dashboard_prompt(
    tokenizer, prompt: str
) -> tuple[list[int] | None, list[str]]:
    """Encode a prompt without letting char-tokenizer misses crash the UI."""

    if hasattr(tokenizer, "stoi"):
        unsupported = sorted(set(prompt).difference(tokenizer.stoi))
        if unsupported:
            return None, unsupported
    return (tokenizer.encode(prompt) if prompt else [0]), []


def show_prompt_warning(unsupported: list[str]) -> None:
    """Explain a tokenizer mismatch in presentation-friendly language."""

    preview = ", ".join(repr(character) for character in unsupported[:8])
    if len(unsupported) > 8:
        preview += f", and {len(unsupported) - 8} more"
    st.warning(
        "This checkpoint uses a character vocabulary learned from tiny Shakespeare. "
        f"Remove unsupported characters ({preview}) and try again."
    )


def quantization_metrics(
    weights: np.ndarray, dequantized: np.ndarray
) -> dict[str, float]:
    """Compute human-readable error metrics for an int8 round trip."""

    residual = weights.astype(np.float64) - dequantized.astype(np.float64)
    mse = float(np.mean(np.square(residual)))
    signal = float(np.mean(np.square(weights.astype(np.float64))))
    sqnr = float("inf") if mse == 0 else 10.0 * math.log10(max(signal, 1e-30) / mse)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": math.sqrt(mse),
        "max_error": float(np.max(np.abs(residual))),
        "sqnr_db": sqnr,
    }


def style_axis(ax: plt.Axes) -> None:
    """Apply the dashboard's dark plotting treatment."""

    ax.set_facecolor("#0d1423")
    ax.tick_params(colors="#aab8ce")
    ax.xaxis.label.set_color("#cbd5e1")
    ax.yaxis.label.set_color("#cbd5e1")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#334155")


def section(kicker: str, title: str, body: str) -> None:
    st.markdown(f'<div class="pico-section">{kicker}</div>', unsafe_allow_html=True)
    st.subheader(title)
    st.caption(body)


def stretch_width(component) -> dict[str, str | bool]:
    """Use the full-width API supported by the installed Streamlit release."""

    width = inspect.signature(component).parameters.get("width")
    if width is not None and isinstance(width.default, str):
        return {"width": "stretch"}
    return {"use_container_width": True}


with st.sidebar:
    st.markdown("### 🧠 PicoLM")
    st.caption("Local model control plane")
    ckpt = st.text_input("Checkpoint", value="out/ckpt.pt")
    if not Path(ckpt).is_file():
        st.warning(
            f"No checkpoint at `{ckpt}`. Train one first:\n\n"
            "`picolm train --out-dir out`"
        )
        st.stop()
    model, tok, device = load(ckpt)
    metrics = load_metrics(ckpt)
    cfg = model.config
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU fallback"
    st.success("Runtime ready")
    st.caption(
        f"{device_name} · {model.get_num_params(non_embedding=False):,} "
        "trainable parameters"
    )
    if metrics:
        st.caption(
            f"best validation loss **{metrics['best_val_loss']:.3f}** · "
            f"{metrics['max_iters']:,} iterations · {metrics['dtype']}"
        )
    st.divider()
    st.markdown("#### Architecture")
    st.caption(
        f"**{cfg.n_layer}** layers · **{cfg.n_head}** heads · "
        f"**{cfg.n_embd}** dimensions\n\n"
        f"context **{cfg.block_size}** · vocabulary **{cfg.vocab_size}** · "
        f"tokenizer **{metrics.get('tokenizer', 'char') if metrics else 'char'}**"
    )
    st.divider()
    st.caption("Built from scratch · no pretrained weights · no Transformers library")


st.markdown(
    f"""
    <section class="pico-hero">
        <div class="pico-kicker">Pico systems lab · live model</div>
        <h1>See a language model <span class="pico-gradient">from the inside.</span></h1>
        <p>PicoLM turns tokenization, causal attention, sampling, training, and
        int8 quantization into an interactive engineering walkthrough.</p>
        <div class="pico-chips">
            <span class="pico-chip">v{_version.__version__}</span>
            <span class="pico-chip">{model.get_num_params(non_embedding=False) / 1e6:.1f}M parameters</span>
            <span class="pico-chip">{cfg.n_layer} transformer layers</span>
            <span class="pico-chip">{device_name}</span>
            <span class="pico-chip">tiny Shakespeare</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

checkpoint_mb = Path(ckpt).stat().st_size / 1e6
metric_columns = st.columns(4)
metric_columns[0].metric(
    "Parameters", f"{model.get_num_params(non_embedding=False) / 1e6:.2f}M"
)
metric_columns[1].metric("Checkpoint", f"{checkpoint_mb:.1f} MB")
metric_columns[2].metric(
    "Best validation loss",
    f"{metrics['best_val_loss']:.3f}" if metrics else "not recorded",
)
metric_columns[3].metric("Runtime", "CUDA" if device == "cuda" else "CPU")

views = ("Overview", "Generate", "Attention", "Sampling", "Quantization", "Training")
view = st.radio("Dashboard view", views, horizontal=True, label_visibility="collapsed")


if view == "Overview":
    section(
        "System map",
        "One small model, the complete language-model pipeline",
        "Use this page to orient the room before opening the deeper visualizations.",
    )
    st.markdown(
        """
        <div class="pico-flow">
            <div class="pico-flow-step">Text<span>raw characters</span></div>
            <div class="pico-flow-step">Tokenizer<span>integer token ids</span></div>
            <div class="pico-flow-step">Transformer<span>causal attention</span></div>
            <div class="pico-flow-step">Logits<span>next-token scores</span></div>
            <div class="pico-flow-step">Sampling<span>generated text</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="pico-card"><strong>01 · From first principles</strong>'
            "<p>Tokenizer, GPT blocks, training loop, KV cache, evaluation, and "
            "quantization are implemented in this repository.</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="pico-card"><strong>02 · Measured, not implied</strong>'
            "<p>The included checkpoint reproduces generation, perplexity, "
            "compression, and decode-speed measurements locally.</p></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="pico-card"><strong>03 · Honest scope</strong>'
            "<p>This is a 10.75M-parameter educational GPT, not a frontier model. "
            "Its value is transparency and systems understanding.</p></div>",
            unsafe_allow_html=True,
        )
    st.markdown("### Suggested three-minute walkthrough")
    st.markdown(
        "1. **Generate** seeded text to prove the checkpoint is live.\n"
        "2. **Attention** show the causal triangle and explain why future tokens are hidden.\n"
        "3. **Quantization** show the 4× storage reduction, then discuss measured error.\n"
        "4. End with the boundary: this demo does **not** claim an optimized int8 kernel speedup."
    )

elif view == "Generate":
    section(
        "Live inference",
        "Generate text from the local checkpoint",
        "Use a fixed seed during a presentation so the output is repeatable.",
    )
    c1, c2 = st.columns([3, 1])
    with c1:
        prompt = st.text_area("Prompt", value="To be, or not to be", height=110)
    with c2:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.8, 0.05)
        top_k = st.slider("Top-k", 0, 200, 40, 5)
        top_p = st.slider("Top-p (0 = off)", 0.0, 1.0, 0.0, 0.05)
        max_tokens = st.slider("Max tokens", 10, 500, 120, 10)
        seed = st.number_input("Seed (0 = random)", 0, 2**31 - 1, 42)
    if st.button("Generate live sample", type="primary", **stretch_width(st.button)):
        if seed:
            torch.manual_seed(seed)
        ids, unsupported = encode_dashboard_prompt(tok, prompt)
        if ids is None:
            show_prompt_warning(unsupported)
            st.stop()
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model.generate(
                idx,
                max_tokens,
                temperature,
                top_k if top_k > 0 else None,
                top_p if top_p > 0 else None,
            )
        st.text_area(
            "Generated output", value=tok.decode(output[0].tolist()), height=250
        )

elif view == "Attention":
    section(
        "Mechanism",
        "Causal self-attention, made visible",
        "Each row is a query token. Each column is the earlier context it may use.",
    )
    prompt = st.text_area(
        "Prompt", value="To be, or not to be", key="attention_prompt", height=80
    )
    control_1, control_2 = st.columns(2)
    layer = control_1.slider("Layer", 0, cfg.n_layer - 1, 0)
    show_all = control_2.toggle("Show every head in this layer", value=False)
    ids, unsupported = encode_dashboard_prompt(tok, prompt)
    if ids is None:
        show_prompt_warning(unsupported)
        st.stop()
    ids = ids[-cfg.block_size :]
    token_count = len(ids)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        _, maps = model.forward_with_attention(idx)
    labels = [tok_label(tok, token_id) for token_id in ids]
    if show_all:
        columns = min(3, cfg.n_head)
        rows = -(-cfg.n_head // columns)
        fig, axes = plt.subplots(
            rows, columns, figsize=(columns * 3.2, rows * 2.8), squeeze=False
        )
        fig.patch.set_alpha(0)
        for head_index in range(cfg.n_head):
            ax = axes[head_index // columns][head_index % columns]
            style_axis(ax)
            ax.imshow(
                maps[layer][0, head_index].cpu().numpy(),
                cmap="magma",
                aspect="auto",
            )
            ax.set_title(f"Head {head_index}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        for head_index in range(cfg.n_head, rows * columns):
            axes[head_index // columns][head_index % columns].axis("off")
        fig.suptitle(f"All attention heads · layer {layer}", color="#f8fafc")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        head = st.slider("Head", 0, cfg.n_head - 1, 0)
        attention = maps[layer][0, head].cpu().numpy()
        fig, ax = plt.subplots(
            figsize=(max(6, token_count * 0.4), max(5, token_count * 0.4))
        )
        fig.patch.set_alpha(0)
        style_axis(ax)
        image = ax.imshow(attention, cmap="magma", aspect="auto")
        ax.set_xticks(range(token_count), labels, rotation=90, fontsize=9)
        ax.set_yticks(range(token_count), labels, fontsize=9)
        ax.set_xlabel("attended token")
        ax.set_ylabel("query token")
        ax.set_title(f"Layer {layer} · head {head}")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("attention weight", color="#cbd5e1")
        colorbar.ax.tick_params(colors="#aab8ce")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

elif view == "Sampling":
    section(
        "Decision surface",
        "Shape the next-token distribution",
        "Temperature changes confidence; top-k and top-p remove the long tail.",
    )
    prompt = st.text_area("Prompt", value="The king", key="sampling_prompt", height=70)
    c1, c2, c3 = st.columns(3)
    temperature = c1.slider("Temperature", 0.0, 2.0, 1.0, 0.05, key="sampling_t")
    top_k = c2.slider("Top-k", 0, 200, 0, 5, key="sampling_k")
    top_p = c3.slider("Top-p (0 = off)", 0.0, 1.0, 0.0, 0.05, key="sampling_p")
    ids, unsupported = encode_dashboard_prompt(tok, prompt)
    if ids is None:
        show_prompt_warning(unsupported)
        st.stop()
    ids = ids[-cfg.block_size :]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, _ = model(idx)
    scaled = logits[0, -1].float() / max(temperature, 1e-6)
    if top_k > 0:
        kept = min(top_k, scaled.size(-1))
        values, _ = torch.topk(scaled, kept)
        scaled[scaled < values[-1]] = float("-inf")
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        keep_mask = torch.zeros_like(scaled, dtype=torch.bool)
        keep_mask.scatter_(0, sorted_indices, ~remove)
        scaled[~keep_mask] = float("-inf")
    probabilities = torch.softmax(scaled, dim=-1)
    top_probabilities, top_ids = torch.topk(probabilities, 20)
    labels = [tok_label(tok, int(token_id)) for token_id in top_ids.cpu()]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_alpha(0)
    style_axis(ax)
    ax.bar(labels, top_probabilities.cpu().numpy(), color="#22d3ee")
    ax.set_ylabel("probability")
    ax.set_xlabel("token")
    ax.set_title("Top-20 next-token probabilities")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    entropy_bits = float(-(probabilities * torch.log2(probabilities + 1e-12)).sum())
    active_tokens = int(torch.isfinite(scaled).sum().item())
    m1, m2 = st.columns(2)
    m1.metric("Distribution entropy", f"{entropy_bits:.3f} bits")
    m2.metric("Candidate tokens", f"{active_tokens} / {cfg.vocab_size}")

elif view == "Quantization":
    section(
        "Compression lab",
        "Watch float32 weights snap onto int8 levels",
        "Symmetric per-tensor quantization stores each matrix as int8 values plus one scale.",
    )
    q_weights, scales, size = load_quantization(ckpt)
    parameter_names = [
        name
        for name, parameter in model.named_parameters()
        if name in q_weights and parameter.dim() >= 2
    ]
    selected = st.selectbox("Weight matrix", parameter_names)
    weights = dict(model.named_parameters())[selected].detach().float().cpu().numpy()
    quantized = q_weights[selected].cpu().numpy()
    dequantized = quantized.astype(np.float32) * scales[selected]
    error = (weights - dequantized).ravel()
    error_metrics = quantization_metrics(weights, dequantized)
    saved_percent = 100.0 * (1.0 - size["int8_bytes"] / size["fp32_bytes"])
    top_metrics = st.columns(5)
    top_metrics[0].metric("FP32 model", f"{size['fp32_bytes'] / 1e6:.1f} MB")
    top_metrics[1].metric("INT8 storage", f"{size['int8_bytes'] / 1e6:.1f} MB")
    top_metrics[2].metric("Compression", f"{size['compression_ratio']:.2f}×")
    top_metrics[3].metric("Space saved", f"{saved_percent:.1f}%")
    top_metrics[4].metric("Layer SQNR", f"{error_metrics['sqnr_db']:.1f} dB")
    chart_1, chart_2 = st.columns(2)
    with chart_1:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        fig.patch.set_alpha(0)
        style_axis(ax)
        ax.hist(weights.ravel(), bins=110, alpha=0.50, label="float32", color="#22d3ee")
        ax.hist(
            dequantized.ravel(),
            bins=110,
            alpha=0.72,
            label="int8 → float",
            color="#a78bfa",
        )
        ax.set_xlabel("weight value")
        ax.set_ylabel("count")
        ax.set_title("Original vs quantized distribution")
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_color("#e2e8f0")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with chart_2:
        flat_weights = weights.ravel()
        flat_dequantized = dequantized.ravel()
        sample_count = min(3_000, flat_weights.size)
        indices = np.linspace(0, flat_weights.size - 1, sample_count, dtype=np.int64)
        sampled_weights = flat_weights[indices]
        sampled_dequantized = flat_dequantized[indices]
        lower = float(min(sampled_weights.min(), sampled_dequantized.min()))
        upper = float(max(sampled_weights.max(), sampled_dequantized.max()))
        fig, ax = plt.subplots(figsize=(7, 3.8))
        fig.patch.set_alpha(0)
        style_axis(ax)
        ax.scatter(
            sampled_weights,
            sampled_dequantized,
            s=8,
            alpha=0.36,
            color="#34d399",
            edgecolors="none",
        )
        ax.plot([lower, upper], [lower, upper], color="#94a3b8", linestyle="--")
        ax.set_xlabel("float32 weight")
        ax.set_ylabel("dequantized int8 weight")
        ax.set_title("Round-trip fidelity · dashed line is perfect")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    detail_1, detail_2, detail_3, detail_4 = st.columns(4)
    detail_1.metric("Scale", f"{scales[selected]:.3e}")
    detail_2.metric("INT8 levels used", f"{len(np.unique(quantized))} / 255")
    detail_3.metric("Mean absolute error", f"{error_metrics['mae']:.3e}")
    detail_4.metric("Maximum error", f"{error_metrics['max_error']:.3e}")
    with st.expander("Inspect individual quantization decisions"):
        rows = min(32, weights.size)
        sample_indices = np.linspace(0, weights.size - 1, rows, dtype=np.int64)
        sample_table = pd.DataFrame(
            {
                "float32": weights.ravel()[sample_indices],
                "int8 code": quantized.ravel()[sample_indices],
                "dequantized": dequantized.ravel()[sample_indices],
                "error": error[sample_indices],
            }
        )
        st.dataframe(sample_table, hide_index=True, **stretch_width(st.dataframe))
    st.info(
        "**Engineering boundary:** this demonstrates storage compression and numerical "
        "round-trip error. PicoLM currently dequantizes weights for evaluation; it does "
        "not claim an optimized int8 inference-kernel speedup."
    )
    with st.expander("How to explain this in 30 seconds"):
        st.markdown(
            "- Find the largest absolute value in one matrix.\n"
            "- Divide by 127 to obtain one shared scale.\n"
            "- Round every weight to an integer from −127 to 127.\n"
            "- Store one byte per weight instead of four; multiply by the scale when needed.\n"
            "- The SQNR and error cards quantify what was lost instead of saying it is ‘free’."
        )

elif view == "Training":
    section(
        "Learning dynamics",
        "From random characters to Shakespeare-like structure",
        "The checkpoint is selected by validation loss, not the final training step.",
    )
    if not metrics:
        st.info("No `metrics.json` was found next to the checkpoint.")
    else:
        eval_interval = metrics.get("eval_interval", 250)
        x_values = [index * eval_interval for index in range(len(metrics["val_loss"]))]
        st.markdown("#### Loss curve")
        st.line_chart(
            pd.DataFrame(
                {"train": metrics["train_loss"], "validation": metrics["val_loss"]},
                index=x_values,
            )
        )
        warmup = metrics.get("warmup_iters", 100)
        maximum_lr = metrics.get("learning_rate", 3e-4)
        minimum_lr = maximum_lr * 0.1
        iterations = metrics.get("max_iters", 2_000)
        schedule_x = list(range(iterations + 1))
        schedule = []
        for iteration in schedule_x:
            if iteration < warmup:
                schedule.append(maximum_lr * (iteration + 1) / warmup)
            else:
                progress = (iteration - warmup) / max(1, iterations - warmup)
                schedule.append(
                    minimum_lr
                    + 0.5
                    * (maximum_lr - minimum_lr)
                    * (1 + math.cos(math.pi * progress))
                )
        st.markdown("#### Cosine learning-rate schedule")
        st.line_chart(pd.DataFrame({"learning rate": schedule}, index=schedule_x))
        if metrics.get("samples"):
            st.markdown("#### Training progression")
            sample_steps = [str(item["step"]) for item in metrics["samples"]]
            selected_step = st.select_slider(
                "Checkpoint step", options=sample_steps, value=sample_steps[-1]
            )
            selected_sample = next(
                item
                for item in metrics["samples"]
                if str(item["step"]) == selected_step
            )
            st.code(selected_sample["text"], language=None)


st.markdown("---")
st.caption(
    "PicoLM Systems Lab · trained on tiny Shakespeare · see MODEL_CARD.md and "
    "REPRODUCIBILITY.md for methods, evidence, and limitations."
)
