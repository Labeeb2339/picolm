"""Run zero-shot HellaSwag evaluation on a PicoLM checkpoint.

Usage::

    python scripts/run_hellaswag.py --ckpt out/ckpt.pt --download --limit 2000

Downloads the HellaSwag validation set (JSONL) on first use, then scores each
example with the model's tokenizer and reports accuracy (chance = 25%).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from picolm.cli import _load_tokenizer
from picolm.hellaswag import evaluate_hellaswag
from picolm.model import GPT

HELLASWAG_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"


def load_examples(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            examples.append({"ctx": ex["ctx"], "endings": ex["endings"], "label": ex["label"]})
    return examples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/hellaswag_val.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="max examples to score")
    ap.add_argument("--download", action="store_true", help="download the val set if missing")
    args = ap.parse_args()

    data = Path(args.data)
    if args.download and not data.exists():
        import urllib.request

        print("downloading HellaSwag validation set ...")
        urllib.request.urlretrieve(HELLASWAG_URL, data)
        print(f"  saved {data} ({data.stat().st_size / 1e6:.1f} MB)")

    if not data.exists():
        print(f"no dataset at {data} — pass --download or place it there manually")
        return 1

    model = GPT.load(args.ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    tok = _load_tokenizer(Path(args.ckpt).parent)

    examples = load_examples(data)
    n = args.limit or len(examples)
    print(f"loaded {len(examples)} examples, scoring {n} on {device} ...")

    result = evaluate_hellaswag(model, tok, examples, torch.device(device), limit=args.limit)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
