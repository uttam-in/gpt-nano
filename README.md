# gpt-nano — build a small GPT from scratch, end to end

A complete, **beginner-friendly** project that takes you from raw text all the
way to a chat model with tool calling that you can publish on Hugging Face.

You train a **tiny (~14M parameter) GPT** from scratch. It's small enough to
train on a **Mac M1** in a reasonable time, and the exact same code runs on an
**NVIDIA GPU (A6000)** when you want to scale up.

> This model is intentionally small. It won't rival ChatGPT — the point is to
> *understand and run every stage of the pipeline yourself*, with code you can
> read in an afternoon.

---

## The stages (each in its own folder)

| Stage | Folder | What you build | What you learn |
|------:|--------|----------------|----------------|
| 0 | [`stage0_data/`](stage0_data/) | Download small text datasets | Where training data comes from |
| 1 | [`stage1_tokenizer/`](stage1_tokenizer/) | Train a BPE tokenizer | How text becomes numbers |
| 2 | [`stage2_pretrain/`](stage2_pretrain/) | Pretrain the GPT on raw text | The core "predict the next token" loop |
| 3 | [`stage3_chat_finetune/`](stage3_chat_finetune/) | Fine-tune it to chat | Turning a text model into an assistant |
| 4 | [`stage4_tool_tuning/`](stage4_tool_tuning/) | Teach it to call tools + MCP server | Tool/function calling and MCP |
| 5 | [`stage5_chat_interface/`](stage5_chat_interface/) | CLI + web chat UIs | Talking to your model |
| 6 | [`stage6_publish/`](stage6_publish/) | Export + push to Hugging Face | Sharing a model with the world |

Shared code lives in [`common/`](common/) (the model, tokenizer, device
selection, chat template). Deep-dive tutorials live in [`docs/`](docs/).

---

## Quick start (Mac M1)

```bash
# 0. One-time setup (creates a virtual environment and installs everything)
bash setup.sh
source .venv/bin/activate

# 1. Run the whole pipeline with one script (downloads data, trains, chats)
bash run_all.sh
```

`run_all.sh` runs a small, fast configuration good for learning. To understand
what each step does, run them one at a time (see the per-stage READMEs).

### Manual, step by step

```bash
source .venv/bin/activate

python stage0_data/download_data.py                 # get text
python stage1_tokenizer/train_tokenizer.py          # train tokenizer
python stage2_pretrain/prepare_dataset.py           # tokenize corpus
python stage2_pretrain/train.py --max-iters 2000    # PRETRAIN (the long step)
python stage2_pretrain/generate.py --prompt "Once upon a time"

python stage3_chat_finetune/build_chat_data.py      # make chat data
python stage3_chat_finetune/train_chat.py           # fine-tune for chat
python stage3_chat_finetune/chat_cli.py             # talk to it

python stage4_tool_tuning/build_tool_data.py        # make tool data
python stage4_tool_tuning/train_tools.py            # fine-tune for tools
python stage5_chat_interface/chat_with_tools.py     # chat + tools

python stage5_chat_interface/web_app.py --ckpt checkpoints/chat.pt  # web UI

python stage6_publish/export_to_hf.py --ckpt checkpoints/tools.pt   # to HF format
python stage6_publish/push_to_hub.py --repo <you>/gpt-nano          # publish
```

---

## What you need

- **Mac M1/M2/M3** (uses the Apple GPU via Metal/MPS), or any **NVIDIA GPU**
  (uses CUDA), or just a **CPU** (slower but works).
- **Python 3.10+**. `setup.sh` uses [`uv`](https://github.com/astral-sh/uv) if
  present, otherwise falls back to `python -m venv`.
- ~2 GB free disk for data, dependencies, and checkpoints.

---

## New to all of this? Start here

Read the docs in order — they assume **no prior ML knowledge**:

1. [`docs/00_fundamentals.md`](docs/00_fundamentals.md) — what a language model *is*
2. [`docs/01_tokenization.md`](docs/01_tokenization.md) — turning text into tokens
3. [`docs/02_architecture.md`](docs/02_architecture.md) — inside the Transformer
4. [`docs/03_pretraining.md`](docs/03_pretraining.md) — the training loop, explained
5. [`docs/04_chat_finetuning.md`](docs/04_chat_finetuning.md) — from text model to chatbot
6. [`docs/05_tools_and_mcp.md`](docs/05_tools_and_mcp.md) — tool calling and MCP
7. [`docs/06_publishing.md`](docs/06_publishing.md) — Hugging Face format & Hub
8. [`docs/07_moving_to_nvidia.md`](docs/07_moving_to_nvidia.md) — scaling up on the A6000
9. [`docs/glossary.md`](docs/glossary.md) — every term, defined simply

---

## Moving to your NVIDIA A6000 later

Nothing in the code is Mac-specific. When you switch machines:

- The device is auto-detected (`common/device.py`), so `cuda` is used
  automatically.
- On CUDA you get **bfloat16 mixed precision** and a **fused optimizer** for
  free — training is much faster.
- With 48 GB of VRAM you can grow the model a lot. Try:
  ```bash
  python stage2_pretrain/train.py --n-layer 12 --n-head 12 --n-embd 768 \
      --block-size 512 --batch-size 64 --max-iters 20000
  ```

See [`docs/07_moving_to_nvidia.md`](docs/07_moving_to_nvidia.md) for details.
