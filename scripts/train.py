#!/usr/bin/env python3
"""Train a PicoLM model without installing the package.

Usage::

    python scripts/train.py --text data/input.txt --out-dir out --max-iters 5000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picolm.config import PICO_CONFIG, ModelConfig
from picolm.training import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="data/input.txt")
    p.add_argument("--out-dir", default="out")
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    text = Path(args.text).read_text(encoding="utf-8")
    train(
        text=text,
        config=ModelConfig(**PICO_CONFIG.__dict__),
        out_dir=args.out_dir,
        device=args.device,
        tokenizer_type=args.tokenizer,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
