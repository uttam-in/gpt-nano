"""
Stage 3c — Chat with your fine-tuned model in the terminal.

This is the first time your model behaves like an assistant: type a message,
it replies, and the conversation history is kept so it has context.

Remember: this is a ~15M-param model trained on synthetic data, so expect
simple, sometimes silly answers. The point is that it follows the chat FORMAT.

Run (after stage 3 training):
    python stage3_chat_finetune/chat_cli.py
    python stage3_chat_finetune/chat_cli.py --temperature 0.7

Commands inside the chat:  /reset  to clear history,  /exit  to quit.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.chat_template import DEFAULT_SYSTEM_PROMPT  # noqa: E402
from common.inference import ChatModel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "..", "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=os.path.join(CKPT_DIR, "chat.pt"))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    print("Loading model ...")
    chat = ChatModel(args.ckpt, args.device)
    print(f"Ready. Talking to {args.ckpt}. Type /exit to quit, /reset to clear.\n")

    messages = [{"role": "system", "content": args.system}]
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print("(history cleared)\n")
            continue

        messages.append({"role": "user", "content": user})

        print("bot> ", end="", flush=True)
        prev = ""
        full = ""
        for full in chat.stream_reply(
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        ):
            # stream_reply yields the cumulative text; print only the new part.
            delta = full[len(prev):]
            print(delta, end="", flush=True)
            prev = full
        print("\n")
        messages.append({"role": "assistant", "content": full})


if __name__ == "__main__":
    main()
