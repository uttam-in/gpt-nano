# 03 — Pretraining: the training loop explained

This covers stage 2 (`stage2_pretrain/`). Pretraining is where the model learns
language by practicing next-token prediction on lots of text.

## From text file to training batches

`prepare_dataset.py` tokenizes every `data/*.txt` file into one long sequence of
token IDs, wraps each document with `<|bos|>`/`<|end|>`, and saves it as raw
`uint16` arrays split into:

- `train.bin` (~90%) — what the model learns from
- `val.bin` (~10%) — held out, used only to *check* progress honestly

We save as raw binary so training can **memory-map** the file — read random
chunks instantly without loading gigabytes into RAM.

### What one training example looks like

We grab a random window of `block_size` (256) tokens as the input `x`, and the
same window shifted right by one as the target `y`:

```
x = [The, cat, sat, on, the]
y = [cat, sat, on, the, mat]
```

So at every position the model is asked "given everything up to here, what's
next?" — 256 predictions from a single 256-token window. Efficient.

A **batch** stacks many such windows (e.g. 32) so the GPU works on them in
parallel.

## The loss: measuring wrongness

For each predicted position we compare the model's probability distribution to
the true next token using **cross-entropy loss**. Intuitively: if the model put
high probability on the correct token, loss is low; if it was confidently wrong,
loss is high.

A useful sanity check: with a vocab of 8192, a totally untrained model has loss
≈ ln(8192) ≈ **9.0** (it's guessing uniformly). You'll see training start near
there and drop. Getting to ~3–4 on this data means it has learned real
structure.

## The optimization loop (`train.py`)

Each iteration:

1. **Get a batch** of (x, y) from `train.bin`.
2. **Forward pass**: run the model, compute the loss.
3. **Backward pass**: `loss.backward()` computes gradients (which way to nudge
   every parameter) automatically.
4. **Optimizer step**: AdamW nudges all parameters a small amount.
5. Occasionally, **evaluate** on train+val and **save a checkpoint** if val loss
   improved.

Three techniques you'll see, and why they matter:

### Gradient accumulation
GPUs have limited memory, so we may only fit a small batch. To simulate a bigger
batch (which stabilizes training), we run several small "micro-batches", sum
their gradients, and only then take one optimizer step. `--grad-accum 4` with
`--batch-size 32` behaves like a batch of 128.

### Learning-rate schedule (warmup + cosine decay)
The **learning rate** controls step size. We:
- **warm up** from ~0 over the first ~100 steps (large steps early can diverge),
- then **cosine-decay** it toward a small value (fine adjustments later).

This is the standard recipe and noticeably improves results.

### Gradient clipping
If a gradient is huge, one step could wreck the model. We cap ("clip") the
gradient norm at 1.0 for safety.

## Mixed precision (CUDA) vs full precision (MPS)

On NVIDIA GPUs we train in **bfloat16** — half the memory and ~2× faster, with
almost no quality loss. On Apple MPS we use **float32** (full precision) because
MPS mixed-precision support is still uneven, and our model is small enough that
memory isn't a problem. `common/device.py` handles this choice for you; you
don't change any code between machines.

## Running it

```bash
python stage2_pretrain/train.py --max-iters 2000 --eval-interval 250
```

Watch the printed `train` and `val` loss drop. When done, the best checkpoint is
at `checkpoints/pretrained.pt`.

Tips:
- More iterations = better text (until it plateaus or overfits — watch val loss).
- If val loss rises while train loss keeps falling, you're **overfitting**; stop
  earlier or add data.
- `--device cpu` forces CPU, which gives clearer error messages when debugging.

## Seeing the payoff

```bash
python stage2_pretrain/generate.py --prompt "Once upon a time"
```

The base model just *continues* text in the style of Shakespeare + simple
stories. It is **not** a chatbot yet — ask it a question and it'll ramble. That's
expected. Making it converse is stage 3.

➡️ Next: [`04_chat_finetuning.md`](04_chat_finetuning.md).
