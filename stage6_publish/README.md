# Stage 6 — Publish to Hugging Face

Convert the from-scratch checkpoint into standard Hugging Face format and upload
it to the Hub.

## Steps

```bash
# 1. Export to HF format (also verifies by reloading + generating)
python stage6_publish/export_to_hf.py --ckpt ../checkpoints/tools.pt --out ../hf_model

# 2. Log in once
huggingface-cli login          # paste a WRITE token from huggingface.co/settings/tokens

# 3. Push
python stage6_publish/push_to_hub.py --repo your-username/gpt-nano
python stage6_publish/push_to_hub.py --repo your-username/gpt-nano --private
```

After pushing, anyone can load it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("your-username/gpt-nano")
tok   = AutoTokenizer.from_pretrained("your-username/gpt-nano")
```

## How the export works

Our model mirrors GPT-2, so `export_to_hf.py` copies weights into a
`GPT2LMHeadModel`, **transposing** the attention/MLP weight matrices (our
`nn.Linear` vs HF's `Conv1D`). It also wraps the tokenizer and writes a Jinja
**chat template** so `apply_chat_template` reproduces our training format.

## Files

- `export_to_hf.py` — convert checkpoint → `hf_model/` (config, safetensors,
  tokenizer, chat template) and verify.
- `push_to_hub.py` — create the repo, write a model card, upload the folder.

## Before you publish

- Check your training data's license permits redistribution (ours does).
- Keep the model card honest about limitations — this is a tiny educational
  model.

📖 Background: [`docs/06_publishing.md`](../docs/06_publishing.md)
