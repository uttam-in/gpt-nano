"""
Stage 1 — Train a Byte-Pair Encoding (BPE) tokenizer.

WHY A TOKENIZER?
    Neural networks work with numbers, not text. A tokenizer converts
        "Hello world"  <->  [15496, 995]
    and back. We train it on our corpus so the most common character sequences
    ("the", "ing", " world") each get a single token, which makes sequences
    short and training efficient.

WHY BPE?
    Byte-Pair Encoding starts from individual bytes and repeatedly merges the
    most frequent adjacent pair into a new token. It's the same family of
    algorithm GPT-2/GPT-3/Llama use. It never fails on unknown words because it
    can always fall back to bytes.

We save the tokenizer in Hugging Face's `tokenizers` JSON format so that later
`transformers` and the Hub can load it directly.

Run (after stage 0):
    python stage1_tokenizer/train_tokenizer.py
    python stage1_tokenizer/train_tokenizer.py --vocab-size 8192
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# Make `common` importable regardless of where we run from.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.chat_template import SPECIAL_TOKENS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(HERE, "tokenizer")  # stage1_tokenizer/tokenizer/


def gather_text_files() -> list[str]:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not files:
        raise SystemExit(
            "No .txt files in data/. Run stage0_data/download_data.py first."
        )
    return files


def train(vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from tokenizers.processors import TemplateProcessing

    files = gather_text_files()
    print(f"[tokenizer] training on {len(files)} file(s):")
    for f in files:
        print(f"    {f}  ({os.path.getsize(f):,} bytes)")

    # A byte-level BPE tokenizer (same style as GPT-2): operates on raw bytes,
    # so it can represent ANY text, including emoji and code.
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,   # reserve our chat tokens as token IDs 0..N
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train(files, trainer)

    # Tell the tokenizer that our special tokens should never be split.
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    os.makedirs(OUT_DIR, exist_ok=True)
    tok_path = os.path.join(OUT_DIR, "tokenizer.json")
    tokenizer.save(tok_path)
    print(f"[done] saved tokenizer -> {tok_path}")
    print(f"       vocab size = {tokenizer.get_vocab_size()}")

    # Also write a tiny meta file the training scripts read to size the model.
    with open(os.path.join(OUT_DIR, "vocab_size.txt"), "w") as f:
        f.write(str(tokenizer.get_vocab_size()))

    # Quick round-trip demo so you can SEE what tokenization does.
    demo = "Hello world! The quick brown fox."
    enc = tokenizer.encode(demo)
    print("\n[demo]")
    print(f"  text   : {demo}")
    print(f"  tokens : {enc.tokens}")
    print(f"  ids    : {enc.ids}")
    print(f"  decoded: {tokenizer.decode(enc.ids)}")

    # Verify special tokens survive a round trip.
    from common.chat_template import USER, EOS
    ids = tokenizer.encode(f"{USER}hi{EOS}").ids
    print(f"\n  '{USER}hi{EOS}' -> ids {ids} -> {tokenizer.decode(ids, skip_special_tokens=False)!r}")

    return tok_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=8192,
                    help="target vocabulary size (default 8192)")
    args = ap.parse_args()
    train(args.vocab_size)
    print("\nNext: python stage2_pretrain/prepare_dataset.py")


if __name__ == "__main__":
    main()
