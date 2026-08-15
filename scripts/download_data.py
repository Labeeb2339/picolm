#!/usr/bin/env python3
"""Download the tiny Shakespeare corpus (used as the default training data).

Usage::

    python scripts/download_data.py
"""

import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

TARGET = Path(__file__).resolve().parents[1] / "data" / "input.txt"


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"already present: {TARGET}")
        return
    print(f"downloading {URL} ...")
    urllib.request.urlretrieve(URL, TARGET)
    print(f"saved {TARGET} ({TARGET.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
