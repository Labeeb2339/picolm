#!/usr/bin/env python3
"""Emit a reproducibility receipt without leaking machine-specific paths.

The receipt binds a run to the Git base commit, the exact visible source tree,
the tracked patch, package versions, hardware, and any reference artifacts that
are present. Generated receipt JSON files under ``reproducibility/receipts`` are
excluded from the source fingerprint to avoid a self-referential hash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reproducibility" / "artifacts.json"
PACKAGES = (
    "torch",
    "numpy",
    "streamlit",
    "matplotlib",
    "pandas",
    "pytest",
    "ruff",
    "build",
    "triton-windows",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_bytes(*args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout


def git_text(*args: str) -> str:
    return git_bytes(*args).decode("utf-8", errors="replace").strip()


def safe_remote(url: str) -> str:
    """Remove any embedded HTTP credentials before recording a remote URL."""

    parts = urlsplit(url)
    if parts.scheme not in {"http", "https", "ssh", "git"} or not parts.hostname:
        return "<non-network-remote-redacted>"
    host = parts.hostname
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def source_manifest() -> tuple[list[dict], str]:
    raw = git_bytes("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    paths = sorted(p.decode("utf-8") for p in raw.split(b"\0") if p)
    entries: list[dict] = []
    aggregate = hashlib.sha256()
    for relative in paths:
        normalized = relative.replace("\\", "/")
        if normalized.startswith("reproducibility/receipts/") and normalized.endswith(
            ".json"
        ):
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        digest = sha256_file(path)
        size = path.stat().st_size
        entries.append({"path": normalized, "bytes": size, "sha256": digest})
        aggregate.update(f"{normalized}\0{size}\0{digest}\n".encode())
    return entries, aggregate.hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def hardware() -> dict:
    import torch

    result = {
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result.update(
            {
                "gpu": properties.name,
                "gpu_memory_bytes": properties.total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
        try:
            driver = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            result["nvidia_driver"] = driver.splitlines()[0]
        except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
            result["nvidia_driver"] = None
    return result


def artifact_evidence() -> tuple[list[dict], bool]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    evidence = []
    all_match = True
    for expected in manifest["artifacts"]:
        path = ROOT / expected["path"]
        observed = {
            "path": expected["path"],
            "present": path.is_file(),
            "expected_sha256": expected["sha256"],
            "expected_bytes": expected["bytes"],
        }
        if path.is_file():
            observed["observed_sha256"] = sha256_file(path)
            observed["observed_bytes"] = path.stat().st_size
            observed["matches"] = (
                observed["observed_sha256"] == expected["sha256"]
                and observed["observed_bytes"] == expected["bytes"]
            )
        else:
            observed["matches"] = False
        all_match = all_match and observed["matches"]
        evidence.append(observed)
    return evidence, all_match


def build_receipt() -> dict:
    files, source_sha256 = source_manifest()
    status = git_text("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    tracked_patch = git_bytes("diff", "--binary", "--no-ext-diff", "HEAD", "--", ".")
    staged_patch = git_bytes(
        "diff", "--binary", "--cached", "--no-ext-diff", "HEAD", "--", "."
    )
    artifacts, artifacts_match = artifact_evidence()
    origin = git_text("remote", "get-url", "origin")
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "project": "PicoLM",
        "git": {
            "head": git_text("rev-parse", "HEAD"),
            "branch": git_text("branch", "--show-current"),
            "origin": safe_remote(origin),
            "dirty": bool(status),
            "status": status,
            "tracked_patch_sha256": sha256_bytes(tracked_patch),
            "staged_patch_sha256": sha256_bytes(staged_patch),
        },
        "source": {
            "fingerprint_sha256": source_sha256,
            "file_count": len(files),
            "files": files,
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "hardware": hardware(),
        },
        "artifacts": artifacts,
        "all_reference_artifacts_match": artifacts_match,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="-", help="JSON path, or - for stdout")
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="exit non-zero unless every reference artifact is present and matches",
    )
    args = parser.parse_args()

    receipt = build_receipt()
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        sys.stdout.write(payload)
    else:
        output = Path(args.output)
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"wrote {output.relative_to(ROOT)}")

    if args.require_artifacts and not receipt["all_reference_artifacts_match"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
