"""Regression tests for portable source receipts."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

RECEIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "repro_receipt.py"
RECEIPT_SPEC = importlib.util.spec_from_file_location(
    "picolm_repro_receipt", RECEIPT_PATH
)
if RECEIPT_SPEC is None or RECEIPT_SPEC.loader is None:
    raise ImportError(f"could not load receipt script from {RECEIPT_PATH}")
receipt = importlib.util.module_from_spec(RECEIPT_SPEC)
RECEIPT_SPEC.loader.exec_module(receipt)


def test_source_manifest_hashes_git_blob_not_checkout_line_endings(
    tmp_path, monkeypatch
) -> None:
    tracked_checkout = b"alpha\r\nbeta\r\n"
    tracked_blob = b"alpha\nbeta\n"
    untracked_bytes = b"draft\r\n"
    (tmp_path / "tracked.txt").write_bytes(tracked_checkout)
    (tmp_path / "draft.txt").write_bytes(untracked_bytes)

    def fake_git_bytes(*args: str) -> bytes:
        if args == ("ls-files", "--cached", "-z"):
            return b"out/evidence/old/receipt.json\0tracked.txt\0"
        if args == ("ls-files", "--others", "--exclude-standard", "-z"):
            return b"draft.txt\0"
        if args == ("show", ":tracked.txt"):
            return tracked_blob
        raise AssertionError(f"unexpected Git call: {args}")

    monkeypatch.setattr(receipt, "ROOT", tmp_path)
    monkeypatch.setattr(receipt, "git_bytes", fake_git_bytes)

    files, fingerprint = receipt.source_manifest()
    by_path = {entry["path"]: entry for entry in files}
    assert "out/evidence/old/receipt.json" not in by_path
    assert by_path["tracked.txt"] == {
        "path": "tracked.txt",
        "basis": "git-index-blob",
        "bytes": len(tracked_blob),
        "sha256": hashlib.sha256(tracked_blob).hexdigest(),
    }
    assert by_path["draft.txt"] == {
        "path": "draft.txt",
        "basis": "untracked-worktree-file",
        "bytes": len(untracked_bytes),
        "sha256": hashlib.sha256(untracked_bytes).hexdigest(),
    }

    aggregate = hashlib.sha256()
    for entry in files:
        aggregate.update(
            (
                f"{entry['path']}\0{entry['basis']}\0{entry['bytes']}\0"
                f"{entry['sha256']}\n"
            ).encode()
        )
    assert fingerprint == aggregate.hexdigest()
