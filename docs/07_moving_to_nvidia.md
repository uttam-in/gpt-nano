# 07 — Moving to your NVIDIA A6000 (scaling up)

You developed everything on a Mac M1. When you move to a machine with an NVIDIA
A6000 (48 GB VRAM), you can train bigger models much faster — **without changing
any code**.

## Nothing to change: device auto-detection

`common/device.py` picks the device in this order: CUDA → MPS → CPU. On the
A6000 box it will automatically choose `cuda`, enable **bfloat16 mixed
precision**, and use the **fused AdamW** optimizer. Just run the same commands.

Verify:

```bash
python common/device.py
# Selected device : cuda
# Description     : CUDA GPU: NVIDIA RTX A6000
# Training dtype  : torch.bfloat16
```

## Setup on the Linux/NVIDIA box

```bash
# Install a CUDA build of PyTorch (check the current command at pytorch.org).
# Example for CUDA 12.x:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt   # the rest
python common/device.py           # should say cuda
```

Everything else (tokenizer, data, training scripts) is identical.

## What the A6000 lets you do

The Mac config is deliberately tiny (~14M params). With 48 GB of VRAM you can go
much bigger. Some concrete steps up:

### 1. A bigger model

```bash
python stage2_pretrain/train.py \
    --n-layer 12 --n-head 12 --n-embd 768 \
    --block-size 512 \
    --batch-size 64 --grad-accum 4 \
    --max-iters 20000 --warmup-iters 500 --eval-interval 1000
```

That's roughly a 120M-parameter model (GPT-2 "small" size) — a huge jump in
quality, still comfortable on an A6000.

Rules of thumb for the size knobs:
- `n_embd` is the "width"; must be divisible by `n_head`.
- `n_layer` is the "depth".
- Params grow roughly with `n_layer × n_embd²`.
- `block_size` (context length) increases memory quadratically via attention —
  raise it deliberately.

### 2. More and better data

Our 2.5M tokens is tiny. On a real GPU, download more:

```bash
python stage0_data/download_data.py --stories 200000
```

or add your own `.txt` files to `data/` and re-run stage 1 + stage 2. For
serious runs, stream a large corpus (e.g. FineWeb, OpenWebText) via the
`datasets` library. More data is usually the highest-leverage change.

### 3. Faster training with `torch.compile`

On CUDA you can wrap the model for a big speedup:

```python
model = torch.compile(model)   # add after model.to(device) in train.py
```

(It's omitted by default because compile support on MPS is immature. On CUDA
it's typically a 1.3–2× speedup.)

### 4. Bigger effective batches

With more VRAM, raise `--batch-size` and lower `--grad-accum` to keep the same
effective batch while running faster (fewer, larger forward passes).

## Memory tuning cheatsheet

If you hit "out of memory" on CUDA:
- lower `--batch-size` (and raise `--grad-accum` to compensate),
- lower `--block-size`,
- shrink the model (`--n-embd`, `--n-layer`).

If VRAM is underused (check `nvidia-smi`), do the opposite — you're leaving speed
on the table.

## Multi-GPU (beyond this project)

This project is single-GPU for clarity. For multiple GPUs you'd reach for
`torchrun` + `DistributedDataParallel`, or a framework like Hugging Face
`accelerate`/`trl`. The model and data code here are compatible with those; the
main change is wrapping the training loop. That's a natural next step once you've
mastered the single-GPU flow.

## The workflow we recommend

1. **Prototype on the Mac** with tiny settings — fast iteration, catch bugs.
2. **Scale on the A6000** — same code, bigger `--n-layer/--n-embd/--max-iters`
   and more data.
3. **Publish** the good checkpoint (stage 6).

➡️ See also: [`glossary.md`](glossary.md).
