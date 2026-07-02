"""
Stage 6b — Publish the exported model to the Hugging Face Hub.

PREREQUISITES:
  1. A free Hugging Face account: https://huggingface.co/join
  2. An access token with "write" permission:
        https://huggingface.co/settings/tokens
  3. Log in once on this machine:
        huggingface-cli login        (paste your token)
     or set the env var HF_TOKEN=hf_xxx before running this script.

This uploads the folder produced by export_to_hf.py (default: ../hf_model) to a
repo like "your-username/gpt-nano". After it finishes, anyone can run:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained("your-username/gpt-nano")

Run:
    python stage6_publish/push_to_hub.py --repo your-username/gpt-nano
    python stage6_publish/push_to_hub.py --repo your-username/gpt-nano --private
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


MODEL_CARD = """---
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
  - gpt2
  - small
  - educational
  - from-scratch
---

# gpt-nano

A tiny (~14M parameter) GPT-2-style language model trained from scratch for
educational purposes. It was pretrained on Tiny Shakespeare + a subset of
TinyStories, then fine-tuned for chat and simple tool calling.

**This is a learning project, not a capable assistant.** Answers are short and
often naive by design — the goal is to demonstrate the full pipeline (tokenizer
-> pretraining -> chat fine-tuning -> tool calling -> publishing) at a scale you
can train on a laptop.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("REPO_ID")
model = AutoModelForCausalLM.from_pretrained("REPO_ID")

messages = [
    {"role": "system", "content": "You are a helpful, concise assistant."},
    {"role": "user", "content": "Hello!"},
]
prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tok(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.7,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
print(tok.decode(out[0], skip_special_tokens=True))
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, required=True, help="e.g. your-username/gpt-nano")
    ap.add_argument("--dir", type=str, default=os.path.join(ROOT, "hf_model"))
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit(f"{args.dir} not found. Run export_to_hf.py first.")

    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")  # falls back to cached CLI login if None

    # Write the model card into the folder so it becomes the repo's README.
    with open(os.path.join(args.dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(MODEL_CARD.replace("REPO_ID", args.repo))

    print(f"[push] creating repo {args.repo} (private={args.private}) ...")
    create_repo(args.repo, private=args.private, exist_ok=True, token=token)

    print(f"[push] uploading {args.dir} ...")
    api = HfApi()
    api.upload_folder(folder_path=args.dir, repo_id=args.repo, token=token)

    print(f"\n[done] https://huggingface.co/{args.repo}")
    print("Anyone can now load it with AutoModelForCausalLM.from_pretrained('%s')" % args.repo)


if __name__ == "__main__":
    main()
