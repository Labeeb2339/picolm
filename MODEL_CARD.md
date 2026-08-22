# PicoLM Model Card

A small GPT-style language model trained from scratch on the tiny Shakespeare
corpus. This document records what the model is, how it was trained, how it was
evaluated, and what it is and is not good for.

---

## Model details

| Field | Value |
|---|---|
| Architecture | Decoder-only transformer (GPT-2 style) |
| Trainable parameters | 10,745,088 total; 10,646,784 excluding learned positional embeddings |
| Layers / heads / dim | 6 / 6 / 384 |
| Context length | 256 tokens |
| Vocab size | 65 (character-level) |
| Tokenizer | Character-level (byte-level BPE also implemented + tested) |
| Activation | GELU (tanh approximation) |
| Positional encoding | Learned (absolute) |
| Weight tying | Input embedding == output head |
| Framework | PyTorch 2.11 (CUDA 12.8) |

## Training

| Field | Value |
|---|---|
| Dataset | tiny Shakespeare (1,115,394 characters) |
| Train / val split | 90 / 10 (first/last contiguous split) |
| Iterations | 2,000 |
| Batch size | 64 |
| Sequence length | 256 |
| Optimizer | AdamW (β = 0.9, 0.95) |
| Weight decay | 0.1 (matrix params only) |
| Learning rate | 3e-4, cosine schedule with 100-step warmup |
| Gradient clipping | 1.0 |
| Precision | bfloat16 mixed precision (AMP + GradScaler) |
| Dropout | 0.2 |
| Early stopping | checkpoint on best validation loss |
| Hardware | NVIDIA RTX 5070 Laptop GPU (8 GB) |
| Wall-clock | ~7.4 minutes |

## Results

Best checkpoint selected by validation loss (step 1,250).

### Perplexity (held-out, 111,540 tokens)

| Model | Loss | Perplexity |
|---|---|---|
| Unigram baseline (most frequent char) | — | 28.43 |
| Bigram baseline (prev → next char) | — | 11.96 |
| **PicoLM** | **1.523** | **4.59** |

PicoLM is **2.61× better than the bigram baseline** and **6.20× better than the
unigram baseline**. Perplexity is `exp(cross-entropy loss)`.

### Quantization

Symmetric per-tensor int8 (`w → round(w / scale)`, `scale = max|w| / 127`):

| Metric | Value |
|---|---|
| fp32 size | 42.98 MB |
| int8 + remaining fp32 size | 10.76 MB |
| Compression | 3.99× |
| Perplexity (fp32) | 4.580 |
| Perplexity (int8) | 4.579 |
| Δ perplexity | −0.023% (same seeded windows) |

No material perplexity change is visible at the reported precision on these
paired windows. The small negative delta is measurement/rounding behaviour, not
evidence that quantization improves quality. This experiment measures int8
storage followed by dequantized PyTorch evaluation; it is not an int8 execution
kernel.

### Decode speed (200 new tokens, RTX 5070)

Repeated synchronized local runs ranged from **0.99× to 1.54×** KV-cache
speedup. This tiny model is dominated by Python and GPU kernel-launch overhead,
so it does not show a stable wall-clock win at 200 tokens. The engineering
benefit is still real: eager decoding recomputes the entire prefix every step,
while the KV-cache reuses prior keys and values. This result is presented as a
mechanism demonstration, not a throughput guarantee.

## Evaluation methodology

- Held-out perplexity uses 100 seeded contiguous batches (64×256). Repeated
  runs select the same windows and produce the same estimate.
- The fp32 and int8 comparison evaluates the exact same seeded windows, so its
  delta is not contaminated by batch-sampling noise.
- CUDA timings synchronize the device before and after each measured decoder.
- Baselines use Laplace (add-1) smoothing and are computed on the identical
  split.
- The loss curve (see `assets/loss.png`) shows the model beginning to overfit
  the small, low-entropy corpus after ~1,500 iterations — train loss falls
  toward 0 while validation loss rises. Early stopping selects the checkpoint
  at the validation-loss minimum rather than the final iteration.

## Known limitations

- **Tiny, low-entropy corpus.** tiny Shakespeare is ~1 MB of repetitive text;
  the model memorizes it beyond ~1,500 iterations. It is a demonstration of the
  architecture, not a general-purpose language model.
- **Character-level tokenization.** The training tokenizer is character-level
  (a deliberate choice for speed and interpretability); the byte-level BPE
  tokenizer is implemented and unit-tested but the demo model is not trained
  with it.
- **Absolute positional embeddings.** GPT-2 style; context beyond 256 tokens is
  out of distribution. The explicit KV-cache path refuses requests beyond this
  window because its eager-parity guarantee does not extend to sliding-window
  decoding.
- **No instruction tuning, no RLHF.** The model only does next-token
  prediction on Shakespeare-style text.
- **Generated text is not factual.** It produces plausible Shakespeare-flavoured
  prose, not information.

## Visualization

`picolm demo` launches an interactive dashboard (Streamlit) that visualizes
the internals: per-layer/head attention maps (the causal triangle), the
next-token probability distribution under temperature/top-k/top-p, fp32-vs-int8
weight histograms, and the training loss + LR curves.

## Reproducibility

The complete environment, command sequence, semantic test guarantees, artifact
hashes, and receipt commands are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
The reference best checkpoint SHA-256 is
`b364afe8bee62f6612642d16394900b0d6c1944ca545e788514cf13b0d636c5a`.

```bash
python scripts/download_data.py
python -m picolm train --text data/input.txt --out-dir out \
    --max-iters 2000 --eval-interval 250 --eval-iters 40 --batch-size 64
python -m picolm eval --ckpt out/ckpt.pt --text data/input.txt
python -m picolm benchmark --ckpt out/ckpt.pt --text data/input.txt
```

Seed is fixed at 42. All metrics (loss curves, samples) are written to
`out/metrics.json`.

## Sample output

*Temperature 0.8, top-k 40, prompt "To be, or not to be":*

> To be, or not to bed,
> His affects of the time modeheal of your consul.
>
> KING EDWARD IV:
> O Clery we speak! what we we that you depeny?
>
> PETER:
> Well, good I shall get us to be a king.
