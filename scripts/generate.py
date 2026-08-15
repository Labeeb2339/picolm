#!/usr/bin/env python3
"""Generate text from a checkpoint without installing the package.

Usage::

    python scripts/generate.py --ckpt out/ckpt.pt --prompt "To be"
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picolm.cli import _load_tokenizer  # noqa: E402
from picolm.inference import generate_kv  # noqa: E402
from picolm.model import GPT  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--use-kv", action="store_true")
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT.load(str(ckpt)).to(device)
    tok = _load_tokenizer(ckpt.parent)

    ids = tok.encode(args.prompt) if args.prompt else [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    if args.use_kv:
        out = generate_kv(
            model, idx, args.max_tokens, args.temperature, args.top_k, args.top_p
        )
    else:
        out = model.generate(idx, args.max_tokens, args.temperature, args.top_k)

    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
