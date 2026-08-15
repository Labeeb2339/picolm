"""Training loop: data prep, gradient updates, evaluation, checkpointing.

Supports single-GPU training with automatic mixed precision (bfloat16) and a
cosine learning-rate schedule with warmup — the standard recipe that makes
small GPTs converge quickly on a consumer GPU.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from picolm.config import ModelConfig
from picolm.model import GPT
from picolm.tokenizer import BPETokenizer, CharTokenizer


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random (x, y) batch of contiguous token sequences."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def _cosine_lr(it: int, warmup: int, decay_iters: int, lr_min: float, lr_max: float) -> float:
    if it < warmup:
        return lr_max * (it + 1) / warmup
    if it > decay_iters:
        return lr_min
    progress = (it - warmup) / max(1, decay_iters - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    """Average loss over ``eval_iters`` random batches on train and val."""
    model.eval()
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out


def train(
    *,
    text: str,
    config: ModelConfig,
    out_dir: str,
    device: str | None = None,
    tokenizer_type: str = "char",
    bpe_vocab_size: int = 512,
    max_iters: int = 5000,
    eval_interval: int = 250,
    eval_iters: int = 100,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    warmup_iters: int = 100,
    weight_decay: float = 1e-1,
    betas: tuple[float, float] = (0.9, 0.95),
    grad_clip: float = 1.0,
    seed: int = 42,
) -> dict:
    """Train a GPT from scratch on ``text`` and return a metrics summary.

    Returns a dict with the loss curves, final model path, and sample outputs.
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = "bfloat16" if device == "cuda" else "float32"
    use_amp = device == "cuda"
    dev = torch.device(device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- tokenize ----------------------------------------------------------
    if tokenizer_type == "bpe":
        tok = BPETokenizer()
        tok.train(text, vocab_size=bpe_vocab_size)
    else:
        tok = CharTokenizer.fit(text)
    config.vocab_size = tok.vocab_size

    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(ids))
    train_data = ids[:n]
    val_data = ids[n:]

    print(
        f"[picolm] vocab={tok.vocab_size} tokens={len(ids)} "
        f"train={len(train_data)} val={len(val_data)} device={device} ({dtype})"
    )

    model = GPT(config).to(dev)
    print(f"[picolm] parameters: {model.get_num_params():,}")

    optimizer = model.configure_optimizers(weight_decay, learning_rate, betas)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    train_losses: list[float] = []
    val_losses: list[float] = []
    samples: list[dict] = []
    t0 = time.time()
    best_val = float("inf")

    for it in range(max_iters + 1):
        lr = _cosine_lr(it, warmup_iters, max_iters, learning_rate * 0.1, learning_rate)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % eval_interval == 0:
            losses = estimate_loss(
                model, train_data, val_data, config.block_size,
                batch_size, eval_iters, dev,
            )
            elapsed = time.time() - t0
            print(
                f"step {it:5d}/{max_iters} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.1f}s"
            )
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])

            # Early stopping: keep the best-by-validation-loss checkpoint.
            if losses["val"] < best_val:
                best_val = losses["val"]
                model.save(str(out_dir / "ckpt.pt"))
                tok.save(out_dir / "tokenizer.json")

            # Generate a short sample to watch the model improve.
            ctx = torch.zeros((1, 1), dtype=torch.long, device=dev)
            with torch.inference_mode():
                gen = model.generate(ctx, max_new_tokens=120, temperature=0.8, top_k=40)
            samples.append({"step": it, "text": tok.decode(gen[0].tolist())})

        # one training step
        xb, yb = get_batch(train_data, config.block_size, batch_size, dev)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

    elapsed = time.time() - t0
    final_losses = estimate_loss(
        model, train_data, val_data, config.block_size, batch_size, eval_iters, dev
    )

    # --- save artifacts ----------------------------------------------------
    # The final iteration may beat every mid-training checkpoint; check once
    # more so ``ckpt.pt`` is always the best-by-val-loss model.
    if final_losses["val"] < best_val:
        best_val = final_losses["val"]
        model.save(str(out_dir / "ckpt.pt"))
        tok.save(out_dir / "tokenizer.json")
    model_path = out_dir / "ckpt.pt"
    # Also keep the very last weights for reference / continued training.
    model.save(str(out_dir / "final.pt"))
    import json

    metrics = {
        "model": config.__dict__,
        "tokenizer": tokenizer_type,
        "device": device,
        "dtype": dtype,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_train_loss": final_losses["train"],
        "final_val_loss": final_losses["val"],
        "best_val_loss": best_val,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "samples": samples,
        "params": model.get_num_params(),
        "elapsed_seconds": round(elapsed, 1),
        "model_path": str(model_path),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(
        f"[picolm] done in {elapsed:.1f}s | final train {final_losses['train']:.4f} "
        f"val {final_losses['val']:.4f} | model -> {model_path}"
    )
    return metrics
