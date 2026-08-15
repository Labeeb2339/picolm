"""Command-line interface.

Examples::

    picolm train --text data/input.txt --out-dir out --max-iters 5000
    picolm generate --ckpt out/ckpt.pt --prompt "To be, or not to be"
    picolm demo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from picolm import _version
from picolm.config import ModelConfig, PICO_CONFIG
from picolm.model import GPT
from picolm.tokenizer import BPETokenizer, CharTokenizer


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="picolm",
        description="GPT-style language model built from scratch in PyTorch.",
    )
    p.add_argument("--version", action="version", version=_version.__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a model from a text corpus")
    t.add_argument("--text", default="data/input.txt")
    t.add_argument("--out-dir", default="out")
    t.add_argument("--max-iters", type=int, default=5000)
    t.add_argument("--eval-interval", type=int, default=250)
    t.add_argument("--eval-iters", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    t.add_argument("--bpe-vocab", type=int, default=512)
    t.add_argument("--device", default=None)
    t.add_argument("--n-layer", type=int, default=None)
    t.add_argument("--n-head", type=int, default=None)
    t.add_argument("--n-embd", type=int, default=None)
    t.add_argument("--block-size", type=int, default=None)
    t.add_argument("--dropout", type=float, default=None)
    t.add_argument("--seed", type=int, default=42)

    g = sub.add_parser("generate", help="generate text from a checkpoint")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--prompt", default="")
    g.add_argument("--max-tokens", type=int, default=200)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=40)
    g.add_argument("--top-p", type=float, default=None)
    g.add_argument("--use-kv", action="store_true", help="use the KV-cache decoder")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--num-samples", type=int, default=1)

    sub.add_parser("demo", help="launch the Streamlit demo")
    return p


def _load_tokenizer(out_dir: Path):
    metrics = out_dir / "metrics.json"
    ttype = "char"
    if metrics.exists():
        ttype = json.loads(metrics.read_text(encoding="utf-8")).get("tokenizer", "char")
    if ttype == "bpe":
        return BPETokenizer.load(out_dir / "tokenizer.json")
    return CharTokenizer.load(out_dir / "tokenizer.json")


def _pick_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cmd_train(args) -> int:
    from picolm.training import train

    cfg = ModelConfig(
        vocab_size=65,
        block_size=args.block_size or PICO_CONFIG.block_size,
        n_layer=args.n_layer or PICO_CONFIG.n_layer,
        n_head=args.n_head or PICO_CONFIG.n_head,
        n_embd=args.n_embd or PICO_CONFIG.n_embd,
        dropout=args.dropout if args.dropout is not None else PICO_CONFIG.dropout,
    )
    text = Path(args.text).read_text(encoding="utf-8")
    train(
        text=text,
        config=cfg,
        out_dir=args.out_dir,
        device=_pick_device(args.device),
        tokenizer_type=args.tokenizer,
        bpe_vocab_size=args.bpe_vocab,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )
    return 0


def _cmd_generate(args) -> int:
    from picolm.inference import generate_kv

    ckpt = Path(args.ckpt)
    model = GPT.load(str(ckpt))
    device = _pick_device(None)
    model.to(device)
    tok = _load_tokenizer(ckpt.parent)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    for i in range(args.num_samples):
        prompt_ids = tok.encode(args.prompt) if args.prompt else [0]
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        if args.use_kv:
            out = generate_kv(
                model, idx, args.max_tokens, args.temperature, args.top_k, args.top_p
            )
        else:
            out = model.generate(
                idx, args.max_tokens, args.temperature, args.top_k
            )
        text = tok.decode(out[0].tolist())
        if args.num_samples > 1:
            print(f"--- sample {i + 1} ---")
        print(text)
        print()
    return 0


def _cmd_demo(_args) -> int:
    import picolm

    demo = Path(picolm.__file__).parent / "demo.py"
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(demo)]
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "train":
        return _cmd_train(args)
    if args.cmd == "generate":
        return _cmd_generate(args)
    if args.cmd == "demo":
        return _cmd_demo(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
