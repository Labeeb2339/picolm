# PicoLM reproducibility and evidence

This document separates code that is present, artifacts that were measured,
and claims that remain hardware-dependent. The committed artifact manifest is
[`reproducibility/artifacts.json`](reproducibility/artifacts.json). Large model
and dataset files stay ignored by Git.

## What the evidence supports

- The default checkpoint is a 6-layer, 6-head, 384-dimensional decoder-only
  transformer with a 256-token context and a 65-character vocabulary.
- It has **10,745,088 total trainable parameters**. The often-used
  non-positional count is **10,646,784**; these are deliberately labelled
  separately.
- The saved best checkpoint, tokenizer, corpus, HellaSwag file, and metrics are
  bound to byte sizes and SHA-256 hashes in the artifact manifest.
- The test suite checks tokenizer round trips and persistence, causal masking,
  eager/KV equality inside the learned context window, sampling filters,
  quantization storage and finite dequantized inference, seeded evaluation,
  checkpoint loading, RoPE/RMSNorm invariants, gradient checkpointing, and the
  HellaSwag scoring harness. CUDA/Triton parity tests are GPU-gated.

It does **not** support claims that PicoLM is factual, competitive with a
pretrained LLM, universally faster with KV caching, or executing matrix
multiplication directly in int8. The int8 experiment stores selected matrix
weights in int8 and dequantizes them for PyTorch evaluation.

One provenance gap is recorded rather than hidden: `out/metrics.json` did not
capture a Git SHA. The first repository commit was created roughly two minutes
after the reference run finished, but that timing is not cryptographic proof of
the exact training source. The artifact hashes are exact; a clean-commit
retraining is still required for end-to-end source-to-checkpoint provenance.

## Reference machine

The local evidence was verified on:

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB |
| NVIDIA driver | 592.15 |
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| cuDNN | 9.19.0 |
| Triton for Windows | 3.7.1.post27 |
| NumPy | 2.4.6 |

The direct-package snapshot is
[`reproducibility/environment-windows-cu128.txt`](reproducibility/environment-windows-cu128.txt).
GPU timing is not portable across drivers, power modes, thermals, or background
load.

## Clean setup

Windows PowerShell, exact evidence environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r reproducibility\environment-windows-cu128.txt
python -m pip install -e . --no-deps
```

Portable CPU verification:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pip check
ruff check .
pytest
```

On this Windows machine, use a repository-local pytest base directory because
the user-wide temporary pytest directory may be inaccessible:

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
python -m pytest -q --basetemp .pytest-tmp\release-cpu
```

## Verify the saved artifacts

On the prepared machine, this hashes every reference file and smoke-loads the
real checkpoint on CPU:

```powershell
.\.venv\Scripts\python.exe scripts\verify_artifacts.py --require-all
```

Generate a machine-readable receipt for the exact current source tree, Git
patch, environment, GPU, and artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\repro_receipt.py `
  --output reproducibility\receipts\local.json `
  --require-artifacts
```

The receipt records both `HEAD` and a SHA-256 fingerprint of all visible source
files. That distinction matters when the worktree is not clean.

## Reproduce the reference training method

The corpus downloader verifies the canonical tiny Shakespeare SHA-256 before
accepting the file.

```powershell
python scripts\download_data.py
python -m picolm train `
  --text data\input.txt `
  --out-dir out `
  --max-iters 2000 `
  --eval-interval 250 `
  --eval-iters 40 `
  --batch-size 64 `
  --lr 3e-4 `
  --seed 42
```

Defaults not repeated on the command line are: character tokenizer, 6 layers,
6 heads, width 384, context 256, dropout 0.2, AdamW betas (0.9, 0.95), weight
decay 0.1, 100 warmup iterations, and gradient clipping at 1.0.

The run is seed-controlled, but CUDA training is not promised to be
bit-identical across software stacks and GPUs. Compare the validation curve and
headline metrics, then create a new artifact manifest for a newly trained
checkpoint; do not claim that a different checkpoint has the reference hash.

## Evaluation and GPU commands

Run the complete sequence and save hash-bound captured logs with one command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_gpu_evidence.ps1
```

Pass `-SkipHellaSwag` for a shorter engineering smoke; that shortened run must
not be presented as a fresh HellaSwag verification. The script writes captured
logs and the matching source/environment receipt under
`out/evidence/<timestamp>/`, then hashes every file into `SHA256SUMS.txt`.
Absolute repository and user-profile prefixes are replaced with `<repo>` and
`<home>` as output is captured so a published bundle does not disclose the
machine's local paths.

Receipt schema v2 hashes canonical Git-index blobs for tracked source files and
raw bytes for untracked files. This keeps the source fingerprint stable across
Git checkout line-ending filters while the recorded clean/dirty state and patch
hashes still expose any source changes outside the index.

The published reference bundle at
[`out/evidence/20260822-235049/`](out/evidence/20260822-235049/) binds the full
run to clean source commit `148a0d28724c505790a8e79c85d61a8c06932f45` and
includes every captured log, `receipt.json`, and their SHA-256 checksums.

Deterministic perplexity and baselines (100 seeded validation batches):

```powershell
python -m picolm eval --ckpt out\ckpt.pt --text data\input.txt --num-batches 100
```

KV-cache timing plus paired fp32/int8 evaluation (the two perplexity passes use
the same seeded windows):

```powershell
python -m picolm benchmark --ckpt out\ckpt.pt --text data\input.txt --max-tokens 200
```

Optional Triton attention correctness and timing:

```powershell
python -m pytest -q tests\test_flash_attn.py
python scripts\bench_attention.py
```

HellaSwag harness check:

```powershell
python scripts\run_hellaswag.py --ckpt out\ckpt.pt --download --limit 2000
```

Keep the captured console output and generated receipt together. A timing number
without its shape, dtype, baseline, synchronization method, hardware, and
source fingerprint is not a reproducible benchmark.
