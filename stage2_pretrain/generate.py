"""
Stage 2c — Generate text from the pretrained model.

This is the payoff of pretraining: give the model a prompt and watch it continue.
Note the base model is NOT a chatbot yet — it just continues text in the style of
its training data (Shakespeare + simple stories). Chatting comes in stage 3.

Run:
    python stage2_pretrain/generate.py --prompt "Once upon a time"
    python stage2_pretrain/generate.py --prompt "ROMEO:" --max-new-tokens 300 --temperature 0.8
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.device import get_device  # noqa: E402
from common.model import GPT, GPTConfig  # noqa: E402
from common.tokenizer import ChatTokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "..", "checkpoints")


def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    config = GPTConfig(**ckpt["model_args"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="\n")
    ap.add_argument("--ckpt", type=str, default=os.path.join(CKPT_DIR, "pretrained.pt"))
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"No checkpoint at {args.ckpt}. Train first (stage2_pretrain/train.py).")

    tok = ChatTokenizer()
    model = load_model(args.ckpt, device)

    ids = [tok.bos_id] + tok.encode(args.prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

    for s in range(args.num_samples):
        out = model.generate(
            x, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            eos_token=tok.eos_id,
        )
        text = tok.decode(out[0].tolist())
        print(f"\n=== sample {s + 1} ===")
        print(text)


if __name__ == "__main__":
    main()
