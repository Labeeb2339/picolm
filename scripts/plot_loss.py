#!/usr/bin/env python3
"""Render the training loss curve and dump final samples from out/metrics.json.

Usage::

    python scripts/plot_loss.py --metrics out/metrics.json --out assets
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="out/metrics.json")
    p.add_argument("--out", default="assets")
    args = p.parse_args()

    m = json.loads(Path(args.metrics).read_text(encoding="utf-8"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Loss curve.
    steps = [i for i in range(len(m["val_loss"]))]
    # eval happens every `eval_interval` steps; reconstruct actual step axis.
    eval_interval = m.get("eval_interval", 300)
    xs = [i * eval_interval for i in steps]

    plt.figure(figsize=(7, 4), dpi=130)
    plt.plot(xs, m["train_loss"], label="train", linewidth=1.5)
    plt.plot(xs, m["val_loss"], label="val", linewidth=1.5)
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.title(
        f"PicoLM training loss — {m.get('params_total', m['params']):,} params, "
        f"{m['max_iters']:,} iters, {m['device']}"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "loss.png")
    print(f"loss curve -> {out / 'loss.png'}")

    # Final sample.
    last = m["samples"][-1]
    (out / "sample_final.txt").write_text(last["text"], encoding="utf-8")
    print(f"final sample -> {out / 'sample_final.txt'}")
    print("--- final sample ---")
    print(last["text"])


if __name__ == "__main__":
    main()
