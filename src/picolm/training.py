"""Training loop: data prep, gradient updates, evaluation, checkpointing.

Supports single-GPU training with automatic mixed precision (bfloat16), a
cosine learning-rate schedule with warmup, gradient checkpointing (recompute
activations in the backward pass to trade compute for memory), distributed
data-parallel training via ``torchrun``, and incremental metrics logging that
survives a crash.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
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
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random (x, y) batch of contiguous token sequences."""
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def _cosine_lr(
    it: int, warmup: int, decay_iters: int, lr_min: float, lr_max: float
) -> float:
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
    grad_checkpoint: bool = False,
) -> dict:
    """Train a GPT from scratch on ``text`` and return a metrics summary.

    ``grad_checkpoint`` recomputes block activations in the backward pass,
    trading ~33% more compute for a ~3.5× reduction in activation memory — the
    lever that lets a small GPU fit a much deeper model. Launch distributed
    runs with ``torchrun --nproc_per_node=N``; ``LOCAL_RANK`` triggers the DDP
    path automatically. DDP is validated on Linux/multi-GPU (NCCL) only:
    Windows torch wheels ship without NCCL and their gloo backend is unstable,
    so the distributed path cannot run on this box.
    """
    torch.manual_seed(seed)
    ddp = int(os.environ.get("LOCAL_RANK", "-1")) != -1
    local_rank = 0
    if ddp:
        import torch.distributed as dist

        # Windows torch wheels ship without NCCL, and their TCPStore has no
        # libuv — fall back to gloo and disable libuv there.
        backend = "nccl" if dist.is_nccl_available() else "gloo"
        os.environ.setdefault("USE_LIBUV", "0")
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.manual_seed(seed + local_rank)  # ranks diverge on data
        device = f"cuda:{local_rank}"
    else:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    is_cuda = dev.type == "cuda"
    is_rank0 = local_rank == 0
    dtype = "bfloat16" if is_cuda else "float32"
    use_amp = is_cuda

    out_dir = Path(out_dir)
    if is_rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
        # A new run must not append evidence to an older run in the same folder.
        (out_dir / "metrics.jsonl").write_text("", encoding="utf-8")

    # --- tokenize ----------------------------------------------------------
    if tokenizer_type == "bpe":
        tok = BPETokenizer()
        tok.train(text, vocab_size=bpe_vocab_size)
    else:
        tok = CharTokenizer.fit(text)
    config.vocab_size = tok.vocab_size
    config.grad_checkpoint = grad_checkpoint

    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(ids))
    train_data = ids[:n]
    val_data = ids[n:]

    if is_rank0:
        print(
            f"[picolm] vocab={tok.vocab_size} tokens={len(ids)} "
            f"train={len(train_data)} val={len(val_data)} device={device} ({dtype}) "
            f"grad_checkpoint={grad_checkpoint} ddp={ddp}"
        )

    raw_model = GPT(config).to(dev)
    n_params_non_positional = raw_model.get_num_params(non_embedding=True)
    n_params_total = raw_model.get_num_params(non_embedding=False)
    model = raw_model
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(raw_model, device_ids=[local_rank])
    if is_rank0:
        print(
            f"[picolm] parameters: {n_params_total:,} total; "
            f"{n_params_non_positional:,} excluding learned positions"
        )

    # Custom GPT methods (configure_optimizers/save/generate) live on the raw
    # module; DDP only wraps the forward/backward path.
    optimizer = raw_model.configure_optimizers(weight_decay, learning_rate, betas)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    train_losses: list[float] = []
    val_losses: list[float] = []
    samples: list[dict] = []
    t0 = time.time()
    best_val = float("inf")
    best_step: int | None = None

    for it in range(max_iters + 1):
        lr = _cosine_lr(it, warmup_iters, max_iters, learning_rate * 0.1, learning_rate)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                config.block_size,
                batch_size,
                eval_iters,
                dev,
            )
            elapsed = time.time() - t0
            if is_rank0:
                print(
                    f"step {it:5d}/{max_iters} | train loss {losses['train']:.4f} | "
                    f"val loss {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.1f}s"
                )
                # Incremental, crash-safe metrics: append one line per eval.
                with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "step": it,
                                "train_loss": round(losses["train"], 4),
                                "val_loss": round(losses["val"], 4),
                                "lr": lr,
                                "elapsed_s": round(elapsed, 1),
                            }
                        )
                        + "\n"
                    )
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])

            # Early stopping: keep the best-by-validation-loss checkpoint.
            if losses["val"] < best_val:
                best_val = losses["val"]
                best_step = it
                if is_rank0:
                    raw_model.save(str(out_dir / "ckpt.pt"))
                    tok.save(out_dir / "tokenizer.json")

            # Generate a short sample to watch the model improve.
            if is_rank0:
                ctx = torch.zeros((1, 1), dtype=torch.long, device=dev)
                with torch.inference_mode():
                    gen = raw_model.generate(
                        ctx, max_new_tokens=120, temperature=0.8, top_k=40
                    )
                samples.append({"step": it, "text": tok.decode(gen[0].tolist())})

        # one training step
        xb, yb = get_batch(train_data, config.block_size, batch_size, dev)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            _logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

    if ddp:
        torch.distributed.barrier()

    elapsed = time.time() - t0
    final_losses = estimate_loss(
        model, train_data, val_data, config.block_size, batch_size, eval_iters, dev
    )

    # --- save artifacts ----------------------------------------------------
    if is_rank0:
        # The final iteration may beat every mid-training checkpoint; check once
        # more so ``ckpt.pt`` is always the best-by-val-loss model.
        if final_losses["val"] < best_val:
            best_val = final_losses["val"]
            best_step = max_iters
            raw_model.save(str(out_dir / "ckpt.pt"))
            tok.save(out_dir / "tokenizer.json")
        model_path = out_dir / "ckpt.pt"
        # Also keep the very last weights for reference / continued training.
        raw_model.save(str(out_dir / "final.pt"))

        metrics = {
            "model": config.__dict__,
            "tokenizer": tokenizer_type,
            "device": device,
            "dtype": dtype,
            "ddp": ddp,
            "world_size": torch.distributed.get_world_size() if ddp else 1,
            "grad_checkpoint": grad_checkpoint,
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
            # ``params`` is retained for compatibility with older plotting code.
            "params": n_params_non_positional,
            "params_non_positional": n_params_non_positional,
            "params_total": n_params_total,
            "best_checkpoint_step": best_step,
            "seed": seed,
            "eval_iters": eval_iters,
            "warmup_iters": warmup_iters,
            "weight_decay": weight_decay,
            "betas": list(betas),
            "grad_clip": grad_clip,
            "data_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "data_characters": len(text),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "elapsed_seconds": round(elapsed, 1),
            "model_path": str(model_path),
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        print(
            f"[picolm] done in {elapsed:.1f}s | final train {final_losses['train']:.4f} "
            f"val {final_losses['val']:.4f} | model -> {model_path}"
        )
    else:
        metrics = {}
    return metrics
