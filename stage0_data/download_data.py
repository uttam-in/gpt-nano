"""
Stage 0 — Download small text datasets for training.

We use two tiny, license-friendly datasets that are perfect for learning:

  1. Tiny Shakespeare (~1 MB): the complete works of Shakespeare concatenated.
     Classic "hello world" of language modeling. Great for pretraining because
     it's small and the model quickly starts producing Shakespeare-ish text.

  2. TinyStories (subset): short, simple stories written with a small vocabulary,
     originally generated to study how small models learn language. We stream a
     subset so we don't download the whole multi-GB dataset.

Everything is saved as plain .txt files under ../data/ so the later stages don't
need internet access.

Run:
    python stage0_data/download_data.py                  # default: both sources
    python stage0_data/download_data.py --stories 20000  # more stories
"""

from __future__ import annotations

import argparse
import os
import urllib.request

# data/ lives at the project root, one level up from this file's folder.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_shakespeare() -> str:
    """Download Tiny Shakespeare to data/shakespeare.txt. Returns the path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, "shakespeare.txt")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"[skip] {out} already exists ({os.path.getsize(out):,} bytes)")
        return out
    print("[download] Tiny Shakespeare ...")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, out)
    print(f"[done] wrote {out} ({os.path.getsize(out):,} bytes)")
    return out


def download_tinystories(num_stories: int = 10000) -> str:
    """Stream a subset of TinyStories to data/tinystories.txt. Returns the path.

    We use the Hugging Face `datasets` library in *streaming* mode, which pulls
    examples one at a time instead of downloading the entire dataset.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, "tinystories.txt")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"[skip] {out} already exists ({os.path.getsize(out):,} bytes)")
        return out

    try:
        from datasets import load_dataset
    except ImportError:
        print("[warn] `datasets` not installed; skipping TinyStories.")
        return out

    print(f"[download] streaming {num_stories:,} TinyStories ...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            if i >= num_stories:
                break
            text = example.get("text", "").strip()
            if text:
                f.write(text)
                f.write("\n\n")  # blank line separates stories
                written += 1
    print(f"[done] wrote {out} ({written:,} stories, {os.path.getsize(out):,} bytes)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", type=int, default=10000,
                    help="number of TinyStories to download (0 to skip)")
    ap.add_argument("--no-shakespeare", action="store_true",
                    help="skip Tiny Shakespeare")
    args = ap.parse_args()

    paths = []
    if not args.no_shakespeare:
        paths.append(download_shakespeare())
    if args.stories > 0:
        paths.append(download_tinystories(args.stories))

    print("\nDatasets ready:")
    for p in paths:
        if os.path.exists(p):
            print(f"  {p}  ({os.path.getsize(p):,} bytes)")
    print("\nNext step: train the tokenizer ->  python stage1_tokenizer/train_tokenizer.py")


if __name__ == "__main__":
    main()
