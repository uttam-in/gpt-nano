# 06 — Publishing to Hugging Face

Covers stage 6 (`stage6_publish/`). We convert our from-scratch model into the
standard Hugging Face format and upload it so anyone can use it.

## Why convert?

We wrote our own `GPT` class, which is great for learning but nobody else's code
knows about it. The Hugging Face `transformers` library is the *lingua franca*
of open models. If we express our model as a standard `GPT2LMHeadModel`, then:

- anyone can load it with `AutoModelForCausalLM.from_pretrained(...)`,
- it works with the whole HF ecosystem (pipelines, text-generation-inference,
  quantization tools, etc.),
- it can live on the Hugging Face Hub with a model card.

## How the conversion works (`export_to_hf.py`)

Our architecture is intentionally GPT-2 compatible, so conversion is mostly a
1:1 copy of weights. The one real difference:

- **We use `nn.Linear`; GPT-2 uses `Conv1D`**, which stores its weight matrix
  transposed. So for the attention and MLP weight matrices we copy the
  **transpose** (`weight.t()`); biases and LayerNorms copy directly.

The script:
1. loads our checkpoint and rebuilds our `GPT`,
2. creates a `GPT2Config` with matching dimensions (and `activation_function=
   "gelu"` to match our exact GELU),
3. copies/transposes every weight into a `GPT2LMHeadModel`,
4. wraps our `tokenizer.json` as a HF `PreTrainedTokenizerFast` with our special
   tokens,
5. writes a **chat template** (Jinja) so users can call
   `tokenizer.apply_chat_template(...)` and get exactly the format we trained on,
6. **verifies** by reloading with plain `transformers` and generating — proving
   the export is faithful.

```bash
python stage6_publish/export_to_hf.py --ckpt checkpoints/tools.pt --out hf_model
```

The output folder `hf_model/` contains `config.json`, `model.safetensors`,
`tokenizer.json`, `tokenizer_config.json`, etc. — a complete, loadable model.

### Sanity-check it yourself

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("hf_model")
model = AutoModelForCausalLM.from_pretrained("hf_model")

msgs = [{"role": "user", "content": "What is 5 + 7?"}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
out = model.generate(**tok(prompt, return_tensors="pt"), max_new_tokens=40,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Pushing to the Hub (`push_to_hub.py`)

1. Make a free account at https://huggingface.co/join
2. Create a **write** token: https://huggingface.co/settings/tokens
3. Log in once: `huggingface-cli login` (paste the token), or set `HF_TOKEN`.
4. Push:

```bash
python stage6_publish/push_to_hub.py --repo your-username/gpt-nano
# or keep it private:
python stage6_publish/push_to_hub.py --repo your-username/gpt-nano --private
```

The script creates the repo, writes a **model card** (`README.md` with license,
tags, and usage), and uploads the folder. When it finishes, your model lives at
`https://huggingface.co/your-username/gpt-nano` and is loadable by name from
anywhere.

## The model card matters

The `README.md` (model card) is how people understand your model: what it is,
what data it saw, its limitations, and how to use it. Ours is honest: it says
this is a tiny educational model that produces simple output. Always document
limitations — it's good practice and required for responsible sharing.

## License note

We tag the model `mit`. Make sure any data you train on permits redistribution.
Tiny Shakespeare (public domain) and TinyStories (permissive) are fine. If you
swap in other data, check its license before publishing.

➡️ Next: [`07_moving_to_nvidia.md`](07_moving_to_nvidia.md).
