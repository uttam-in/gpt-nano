# Stage 1 — Tokenizer

Train a byte-level BPE tokenizer that converts text ↔ token IDs, with our chat
special tokens reserved.

## What it does

`train_tokenizer.py`:
- reads all `../data/*.txt`,
- trains a byte-level BPE tokenizer (vocab 8192 by default),
- reserves the special tokens from `common/chat_template.py`,
- saves `tokenizer/tokenizer.json` in **Hugging Face format**,
- prints a round-trip demo so you can see tokenization in action.

## Run

```bash
python stage1_tokenizer/train_tokenizer.py
python stage1_tokenizer/train_tokenizer.py --vocab-size 16384
```

## Output

- `tokenizer/tokenizer.json` — the tokenizer (used by every later stage).
- `tokenizer/vocab_size.txt` — the vocab size (sets the model's `vocab_size`).

## Why 8192?

Small enough to keep the model tiny, large enough that common words are single
tokens. The tokenizer's vocab size becomes the model's input/output size.

📖 Background: [`docs/01_tokenization.md`](../docs/01_tokenization.md)

➡️ Next: [`stage2_pretrain/`](../stage2_pretrain/)
