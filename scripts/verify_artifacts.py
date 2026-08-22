#!/usr/bin/env python3
"""Verify PicoLM reference artifacts and smoke-load the checkpoint on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "reproducibility" / "artifacts.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_inside_repo(relative: str) -> Path:
    target = (ROOT / relative).resolve()
    if target != ROOT and ROOT not in target.parents:
        raise ValueError(f"artifact path leaves the repository: {relative}")
    return target


def smoke_checkpoint(manifest: dict) -> list[str]:
    import torch

    from picolm.cli import _load_tokenizer
    from picolm.model import GPT

    checkpoint = ROOT / "out" / "ckpt.pt"
    model = GPT.load(str(checkpoint), map_location="cpu").eval()
    tokenizer = _load_tokenizer(checkpoint.parent)
    expected = manifest["model"]

    total = sum(parameter.numel() for parameter in model.parameters())
    non_positional = model.get_num_params(non_embedding=True)
    checks = {
        "total trainable parameter count": total
        == expected["total_trainable_parameters"],
        "non-positional parameter count": non_positional
        == expected["non_positional_parameters"],
        "vocabulary size": tokenizer.vocab_size == expected["vocabulary_size"],
        "context length": model.config.block_size == expected["context_length"],
    }

    ids = tokenizer.encode("To be")
    x = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        logits, loss = model(x)
    checks["CPU forward shape"] = tuple(logits.shape) == (
        1,
        len(ids),
        expected["vocabulary_size"],
    )
    checks["CPU forward finite"] = bool(torch.isfinite(logits).all()) and loss is None

    failures = [name for name, passed in checks.items() if not passed]
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="treat locally absent ignored artifacts as failures",
    )
    parser.add_argument(
        "--no-smoke", action="store_true", help="skip loading the checkpoint on CPU"
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    missing: list[str] = []
    for expected in manifest["artifacts"]:
        path = resolve_inside_repo(expected["path"])
        if not path.is_file():
            missing.append(expected["path"])
            print(f"MISS  {expected['path']} (ignored reference artifact)")
            continue
        digest = sha256_file(path)
        size = path.stat().st_size
        passed = digest == expected["sha256"] and size == expected["bytes"]
        print(f"{'PASS' if passed else 'FAIL'}  {expected['path']}")
        if not passed:
            failures.append(expected["path"])

    checkpoint_present = (ROOT / "out" / "ckpt.pt").is_file()
    if checkpoint_present and not args.no_smoke:
        failures.extend(smoke_checkpoint(manifest))
    elif not checkpoint_present and args.require_all:
        failures.append("checkpoint smoke test")

    if args.require_all:
        failures.extend(missing)
    if failures:
        print("verification failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("artifact verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
