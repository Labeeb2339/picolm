#!/usr/bin/env python3
"""Download the tiny Shakespeare corpus (used as the default training data).

Usage::

    python scripts/download_data.py
"""

import hashlib
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

TARGET = Path(__file__).resolve().parents[1] / "data" / "input.txt"
EXPECTED_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> None:
    observed = sha256_file(path)
    if observed != EXPECTED_SHA256:
        raise RuntimeError(
            f"unexpected tiny Shakespeare SHA-256 for {path}: {observed}"
        )


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        verify(TARGET)
        print(f"already present and verified: {TARGET} ({EXPECTED_SHA256})")
        return
    print(f"downloading {URL} ...")
    urllib.request.urlretrieve(URL, TARGET)
    verify(TARGET)
    print(
        f"saved and verified {TARGET} ({TARGET.stat().st_size:,} bytes, "
        f"sha256={EXPECTED_SHA256})"
    )


if __name__ == "__main__":
    main()
