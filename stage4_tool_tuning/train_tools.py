"""
Stage 4b — Fine-tune for tool calling.

We start from the chat model (checkpoints/chat.pt) so it already knows the
conversation format, and teach it the tool-calling pattern on top. Under the
hood this reuses the SAME loss-masked SFT routine as stage 3.

Run (after stage 3 + build_tool_data.py):
    python stage4_tool_tuning/train_tools.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stage3_chat_finetune.train_chat import run_sft  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "..", "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=os.path.join(HERE, "tool_data.jsonl"))
    # Start from the chat model if it exists, else fall back to pretrained.
    default_init = os.path.join(CKPT_DIR, "chat.pt")
    if not os.path.exists(default_init):
        default_init = os.path.join(CKPT_DIR, "pretrained.pt")
    ap.add_argument("--init", type=str, default=default_init)
    ap.add_argument("--out", type=str, default=os.path.join(CKPT_DIR, "tools.pt"))
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
        device_str=args.device, seed=args.seed, label="tools",
    )
    print("\nNext: python stage4_tool_tuning/mcp_server.py  (serve the tools over MCP)")
    print("  or: python stage5_chat_interface/chat_with_tools.py  (chat + tools)")


if __name__ == "__main__":
    main()
