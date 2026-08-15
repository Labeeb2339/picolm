# PicoLM Model Card

A small GPT-style language model trained from scratch on the tiny Shakespeare
corpus. This document records what the model is, how it was trained, how it was
evaluated, and what it is and is not good for.

---

## Model details

| Field | Value |
|---|---|
| Architecture | Decoder-only transformer (GPT-2 style) |
| Parameters | 10,646,784 (~10.6M) |
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
| Train / val split | 90 / 10 (stratified contiguous split) |
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
| **PicoLM** | **1.517** | **4.56** |

PicoLM is **2.62× better than the bigram baseline** and **6.24× better than the
unigram baseline**. Perplexity is `exp(cross-entropy loss)`.

### Quantization

Symmetric per-tensor int8 (`w → round(w / scale)`, `scale = max|w| / 127`):

| Metric | Value |
|---|---|
| fp32 size | 42.98 MB |
| int8 size | 10.74 MB |
| Compression | 4.0× |
| Perplexity (fp32) | 4.598 |
| Perplexity (int8) | 4.595 |
| Δ perplexity | −0.07% (within noise) |

Weight quantization is essentially free for this model — expected, since int8
has plenty of dynamic range for these weight magnitudes.

### Decode speed (200 new tokens, RTX 5070)

| Method | tokens/sec | notes |
|---|---|---|
| Eager (recompute full prefix) | 148.6 | O(T³) total attention |
| KV-cache (reuse past K/V) | 179.3 | O(T²) total attention |

Observed speedup is **1.21×** at 200 tokens. The measured gain understates the
asymptotic benefit: for a 10.6M-param model at short sequence lengths, GPU
kernel-launch overhead dominates attention compute. The eager decoder
recomputes the entire prefix every step (O(T³) total), while the KV-cache
decoder attends only the new token against the cached history (O(T²) total),
so the gap widens rapidly with sequence length and model size.

## Evaluation methodology

- Held-out perplexity is computed over 100 random contiguous batches (64×256),
  so the estimate is stable and free of train/val leakage.
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
  out of distribution (the KV-cache clamps the positional index).
- **No instruction tuning, no RLHF.** The model only does next-token
  prediction on Shakespeare-style text.
- **Generated text is not factual.** It produces plausible Shakespeare-flavoured
  prose, not information.

## Reproducibility

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
