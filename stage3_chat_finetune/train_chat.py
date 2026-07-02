"""
Stage 3b — Fine-tune the pretrained model to follow the chat format.

We start from the stage-2 checkpoint (checkpoints/pretrained.pt) and keep
training, but now on CONVERSATIONS with loss masking (see common/sft.py). The
model learns to: read <|user|>...<|end|><|assistant|>, then produce a reply and
emit <|end|> to stop.

Note the one-token SHIFT: the model predicts token t+1 from tokens up to t, so
we feed input_ids[:, :-1] and compare against labels[:, 1:].

Run (after stage 2 + build_chat_data.py):
    python stage3_chat_finetune/train_chat.py
    python stage3_chat_finetune/train_chat.py --epochs 3 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.device import autocast_context, describe, get_device  # noqa: E402
from common.model import GPT, GPTConfig  # noqa: E402
from common.sft import ConversationDataset, make_collate_fn  # noqa: E402
from common.tokenizer import ChatTokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "..", "checkpoints")


def load_conversations(path: str) -> list[list[dict]]:
    convos = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                convos.append(json.loads(line)["messages"])
    return convos


def run_sft(
    data_path: str,
    init_ckpt: str,
    out_ckpt: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    device_str: str | None,
    seed: int,
    label: str = "chat",
):
    """Shared fine-tuning routine (also called by stage 4 for tool tuning)."""
    torch.manual_seed(seed)
    device = get_device(device_str)
    print(f"[{label}] device: {describe(device)}")

    if not os.path.exists(init_ckpt):
        raise SystemExit(f"No init checkpoint at {init_ckpt}. Run the previous stage first.")

    tok = ChatTokenizer()
    ckpt = torch.load(init_ckpt, map_location=device)
    config = GPTConfig(**ckpt["model_args"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"[{label}] loaded {init_ckpt} ({model.num_params():,} params)")

    convos = load_conversations(data_path)
    dataset = ConversationDataset(convos, tok, config.block_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=make_collate_fn(tok.pad_id),
    )
    print(f"[{label}] {len(dataset):,} training conversations, "
          f"{len(loader)} batches/epoch")

    optimizer = model.configure_optimizers(
        weight_decay=weight_decay, learning_rate=lr,
        betas=(0.9, 0.95), device_type=device.type,
    )

    model.train()
    t0 = time.time()
    step = 0
    for epoch in range(epochs):
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            # One-token shift: predict token t+1 from tokens up to t.
            x = input_ids[:, :-1].contiguous()
            y = labels[:, 1:].contiguous()

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                _, loss = model(x, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if step % 50 == 0:
                dt = time.time() - t0
                print(f"epoch {epoch} step {step:5d} | loss {loss.item():.4f} | {dt:.1f}s")
            step += 1

    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_args": config.__dict__,
        "config": config.__dict__,
    }, out_ckpt)
    print(f"\n[{label}] done. saved -> {out_ckpt}")
    return out_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=os.path.join(HERE, "chat_data.jsonl"))
    ap.add_argument("--init", type=str, default=os.path.join(CKPT_DIR, "pretrained.pt"))
    ap.add_argument("--out", type=str, default=os.path.join(CKPT_DIR, "chat.pt"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    run_sft(
        data_path=args.data, init_ckpt=args.init, out_ckpt=args.out,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        device_str=args.device, seed=args.seed, label="chat",
    )
    print("\nNext: python stage3_chat_finetune/chat_cli.py  (talk to your model)")


if __name__ == "__main__":
    main()
