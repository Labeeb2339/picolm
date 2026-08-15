"""Streamlit demo for PicoLM.

Launch with ``picolm demo`` (or ``streamlit run src/picolm/demo.py``). Load a
checkpoint, watch the loss curve, and generate text with live sampling knobs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import streamlit as st

from picolm import _version
from picolm.cli import _load_tokenizer
from picolm.inference import generate_kv
from picolm.model import GPT

st.set_page_config(page_title="PicoLM", page_icon="🧠", layout="wide")

st.title("🧠 PicoLM")
st.caption(
    f"GPT-style language model trained from scratch · v{_version.__version__} · "
    "hand-written BPE + transformer + KV-cache inference"
)


@st.cache_resource
def load_model(ckpt_path: str):
    model = GPT.load(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tok = _load_tokenizer(Path(ckpt_path).parent)
    return model, tok, device


with st.sidebar:
    st.header("Model")
    ckpt = st.text_input("Checkpoint path", value="out/ckpt.pt")

    if Path(ckpt).exists():
        model, tok, device = load_model(ckpt)
        st.success(f"Loaded · device={device} · {model.get_num_params():,} params")

        # Loss curve.
        metrics_path = Path(ckpt).parent / "metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            if m.get("val_loss"):
                st.subheader("Validation loss")
                st.line_chart(
                    {"val": m["val_loss"], "train": m["train_loss"]}
                )

        st.header("Sampling")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.8, 0.05)
        top_k = st.slider("Top-k", 0, 200, 40, 5)
        top_p = st.slider("Top-p (0 = off)", 0.0, 1.0, 0.0, 0.05)
        use_kv = st.checkbox("Use KV-cache decoder", value=True)
        max_tokens = st.slider("Max tokens", 10, 1000, 200, 10)
        seed = st.number_input("Seed (0 = random)", 0, 2**31 - 1, 0)

    else:
        st.warning(f"No checkpoint at `{ckpt}`. Train one first: "
                   "`picolm train --out-dir out`")

st.header("Generate")
prompt = st.text_area("Prompt", value="To be, or not to be", height=100)

if st.button("Generate", type="primary") and Path(ckpt).exists():
    if seed:
        torch.manual_seed(seed)
    ids = tok.encode(prompt) if prompt else [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        if use_kv:
            out = generate_kv(
                model, idx, max_tokens, temperature,
                top_k if top_k > 0 else None, top_p if top_p > 0 else None,
            )
        else:
            out = model.generate(
                idx, max_tokens, temperature, top_k if top_k > 0 else None
            )
    text = tok.decode(out[0].tolist())
    st.text_area("Output", value=text, height=300)

st.markdown("---")
st.caption("Trained on tiny Shakespeare. Educational project — a from-scratch "
           "reimplementation of the GPT architecture.")
