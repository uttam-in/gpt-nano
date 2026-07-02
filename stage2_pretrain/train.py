"""
Stage 2b — Pretrain the GPT model on the tokenized corpus.

WHAT "PRETRAINING" MEANS:
    We show the model billions... okay, in our small case millions of tokens and
    ask it to predict the next token at every position. Doing this over and over
    teaches it grammar, facts, and style. This is the expensive, foundational
    step; everything after (chat, tools) is cheap fine-tuning on top.

The loop is intentionally small and readable. Key ideas you'll see:
  - get_batch(): grab a random chunk of the token stream.
  - cosine LR schedule with warmup: standard recipe for stable training.
  - gradient accumulation: simulate a big batch on a small GPU.
  - periodic eval + checkpoint saving.

Runs on Mac M1 (MPS) or NVIDIA (CUDA) unchanged — see common/device.py.

Run:
    python stage2_pretrain/train.py                       # sensible small defaults
    python stage2_pretrain/train.py --max-iters 3000 --eval-interval 250
    python stage2_pretrain/train.py --device cpu          # force CPU for debugging
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.device import autocast_context, describe, get_device  # noqa: E402
from common.model import GPT, GPTConfig  # noqa: E402
from common.tokenizer import ChatTokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_BIN = os.path.join(HERE, "data_bin")
CKPT_DIR = os.path.join(HERE, "..", "checkpoints")


def get_batch(split: str, block_size: int, batch_size: int, device: torch.device):
    """Load a random batch of (input, target) pairs from the .bin files.

    We re-open the memmap each call; it's cheap and avoids stale file handles.
    targets are just inputs shifted right by one — "predict the next token".
    """
    path = os.path.join(DATA_BIN, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.type == "cuda":
        # pin_memory + non_blocking overlaps the CPU->GPU copy with compute.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, block_size, batch_size, device, eval_iters=50):
    """Average the loss over a few random batches for train and val splits."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, block_size, batch_size, device)
            with autocast_context(device):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Cosine learning-rate schedule with linear warmup."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def main():
    ap = argparse.ArgumentParser()
    # Model size (defaults = ~15M params; matches common/model.py defaults)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    # Optimization
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="accumulate this many batches before an optimizer step")
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--warmup-iters", type=int, default=100)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=50)
    # Misc
    ap.add_argument("--device", type=str, default=None, help="cpu|mps|cuda (auto if unset)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=str, default=os.path.join(CKPT_DIR, "pretrained.pt"))
    ap.add_argument("--init-from", type=str, default=None,
                    help="resume from an existing checkpoint .pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    print(f"[train] device: {describe(device)}")

    tok = ChatTokenizer()

    # Build the model config from CLI args + tokenizer vocab size.
    if args.init_from and os.path.exists(args.init_from):
        print(f"[train] resuming from {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=device)
        config = GPTConfig(**ckpt["model_args"])
        model = GPT(config)
        model.load_state_dict(ckpt["model"])
        start_iter = ckpt.get("iter", 0)
    else:
        config = GPTConfig(
            vocab_size=tok.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        )
        model = GPT(config)
        start_iter = 0

    model.to(device)
    print(f"[train] model params: {model.num_params():,}")

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=device.type,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_val = float("inf")
    t0 = time.time()

    for it in range(start_iter, args.max_iters + 1):
        # Set the learning rate for this step.
        lr = get_lr(it, args.warmup_iters, args.max_iters, args.learning_rate, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- Evaluation + checkpoint ----
        if it % args.eval_interval == 0:
            losses = estimate_loss(model, config.block_size, args.batch_size, device, args.eval_iters)
            dt = time.time() - t0
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {lr:.2e} | {dt:.1f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model": model.state_dict(),
                    "model_args": vars(config) if hasattr(config, "__dict__") else config.__dict__,
                    "iter": it,
                    "best_val": best_val,
                    "config": config.__dict__,
                }, args.out)
                print(f"         saved checkpoint -> {args.out} (val {best_val:.4f})")

        if it == args.max_iters:
            break

        # ---- One optimization step with gradient accumulation ----
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            x, y = get_batch("train", config.block_size, args.batch_size, device)
            with autocast_context(device):
                _, loss = model(x, y)
                loss = loss / args.grad_accum  # scale so accumulation = averaging
            loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    print(f"\n[done] best val loss: {best_val:.4f}")
    print(f"       checkpoint: {args.out}")
    print("\nNext: python stage2_pretrain/generate.py")


if __name__ == "__main__":
    main()
