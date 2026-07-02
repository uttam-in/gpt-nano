# Stage 3 — Chat fine-tuning

Turn the base model into an assistant by fine-tuning on conversations with
**loss masking** (train only on the assistant's replies).

## Steps

```bash
# 1. Build a synthetic chat dataset (greetings, arithmetic, echo, facts)
python stage3_chat_finetune/build_chat_data.py --n 4000

# 2. Fine-tune the pretrained model for chat
python stage3_chat_finetune/train_chat.py --epochs 3

# 3. Talk to it
python stage3_chat_finetune/chat_cli.py
```

Inside the chat: `/reset` clears history, `/exit` quits.

## How it works

- `build_chat_data.py` writes `chat_data.jsonl`, one conversation per line as
  `{"messages": [{"role","content"}, ...]}`.
- `train_chat.py` loads `../checkpoints/pretrained.pt`, applies the chat template
  + loss mask (`common/sft.py`), and saves `../checkpoints/chat.pt`.
- `chat_cli.py` streams replies token-by-token using `common/inference.py`.

## Expectations

This is a ~14M model on synthetic data — answers are simple and sometimes wrong.
The point is that it **follows the chat format**: reads your message, replies,
and stops. For real quality, swap in a human instruction dataset (same format).

## Swapping in real data

Replace `chat_data.jsonl` with any file of `{"messages":[...]}` lines (e.g. from
Dolly/OpenAssistant/Alpaca converted to this schema). Nothing else changes.

📖 Background: [`docs/04_chat_finetuning.md`](../docs/04_chat_finetuning.md)

➡️ Next: [`stage4_tool_tuning/`](../stage4_tool_tuning/)
