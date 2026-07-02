# Stage 2 — Pretraining

Train the GPT from scratch on raw text via next-token prediction. This is the
foundational (and longest) step.

## Steps

```bash
# 1. Tokenize the corpus into train.bin / val.bin
python stage2_pretrain/prepare_dataset.py

# 2. Pretrain (watch train/val loss drop from ~9.0)
python stage2_pretrain/train.py --max-iters 2000 --eval-interval 250

# 3. Generate text from the base model
python stage2_pretrain/generate.py --prompt "Once upon a time"
```

## Key options for `train.py`

| Flag | Meaning | Default |
|------|---------|---------|
| `--n-layer/--n-head/--n-embd` | model size | 6 / 6 / 384 (~14M params) |
| `--block-size` | context length | 256 |
| `--batch-size` / `--grad-accum` | batch, gradient accumulation | 32 / 4 |
| `--learning-rate` | peak LR | 3e-4 |
| `--max-iters` / `--warmup-iters` | training length / warmup | 2000 / 100 |
| `--eval-interval` | eval + checkpoint every N iters | 250 |
| `--device` | `cpu`/`mps`/`cuda` (auto if unset) | auto |
| `--init-from` | resume from a checkpoint | — |

Output: `../checkpoints/pretrained.pt` (best validation loss).

## What to expect

- Loss starts near **ln(8192) ≈ 9.0** (random guessing) and drops.
- The base model **continues** text; it is **not** a chatbot yet (that's stage 3).
- More iters + more data = better text, until val loss plateaus/rises.

## Files

- `prepare_dataset.py` — tokenize `data/*.txt` → `data_bin/{train,val}.bin`
- `train.py` — the training loop (LR schedule, grad accumulation, checkpoints)
- `generate.py` — sample text from a checkpoint

📖 Background: [`docs/02_architecture.md`](../docs/02_architecture.md),
[`docs/03_pretraining.md`](../docs/03_pretraining.md)

➡️ Next: [`stage3_chat_finetune/`](../stage3_chat_finetune/)
