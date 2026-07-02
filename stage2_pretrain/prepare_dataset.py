"""
Stage 2a — Tokenize the raw text corpus into compact binary files.

We convert all of data/*.txt into one long stream of token IDs, then split it
into a training set and a small validation set. The IDs are saved as raw
uint16 arrays (.bin files) because:

  - They load instantly with numpy's memory-mapping (no RAM blow-up).
  - The training loop can grab random chunks with zero parsing overhead.

Output (in stage2_pretrain/data_bin/):
    train.bin   ~90% of tokens
    val.bin     ~10% of tokens

Run:
    python stage2_pretrain/prepare_dataset.py
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.chat_template import BOS, EOS  # noqa: E402
from common.tokenizer import ChatTokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(HERE, "data_bin")


def main(val_fraction: float = 0.1):
    tok = ChatTokenizer()
    if tok.vocab_size > 65535:
        raise SystemExit("vocab_size > 65535 won't fit in uint16; use uint32.")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not files:
        raise SystemExit("No data/*.txt. Run stage0_data/download_data.py first.")

    print(f"[prepare] tokenizing {len(files)} file(s) ...")
    all_ids: list[int] = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # Wrap each document with BOS/EOS so the model learns where things start/end.
        ids = [tok.bos_id] + tok.encode(text) + [tok.eos_id]
        all_ids.extend(ids)
        print(f"    {os.path.basename(path):20s} -> {len(ids):,} tokens")

    ids = np.array(all_ids, dtype=np.uint16)
    n_val = int(len(ids) * val_fraction)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    os.makedirs(OUT_DIR, exist_ok=True)
    train_ids.tofile(os.path.join(OUT_DIR, "train.bin"))
    val_ids.tofile(os.path.join(OUT_DIR, "val.bin"))

    print(f"\n[done] total tokens : {len(ids):,}")
    print(f"       train.bin    : {len(train_ids):,} tokens")
    print(f"       val.bin      : {len(val_ids):,} tokens")
    print(f"       vocab size   : {tok.vocab_size}")
    print("\nNext: python stage2_pretrain/train.py")


if __name__ == "__main__":
    main()
