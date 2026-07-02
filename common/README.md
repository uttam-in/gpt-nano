# common/ — shared code

Reusable modules imported by every stage. Reading these top-to-bottom is the
fastest way to understand the whole system.

| File | What it is |
|------|-----------|
| `device.py` | Auto-selects CUDA / MPS / CPU and the right dtype. The reason the same code runs on Mac and NVIDIA. |
| `model.py` | The GPT model, from scratch (~250 lines). Attention, MLP, blocks, generation. |
| `chat_template.py` | The special tokens and how a conversation is rendered into a training/inference string (with trainable spans). |
| `tokenizer.py` | Thin wrapper (`ChatTokenizer`) around the trained `tokenizer.json`. |
| `sft.py` | Supervised fine-tuning helpers: turns conversations into loss-masked `(input_ids, labels)` batches. |
| `inference.py` | `ChatModel`: load a checkpoint and stream assistant replies. |
| `tool_loop.py` | The agent loop: model → tool call → run tool → result → final answer. |

Suggested reading order for learners: `device.py` → `model.py` →
`chat_template.py` → `tokenizer.py` → `sft.py` → `inference.py` → `tool_loop.py`.

Each file has a module docstring explaining its purpose and can be run directly
(`python common/model.py`, etc.) for a quick self-test.
