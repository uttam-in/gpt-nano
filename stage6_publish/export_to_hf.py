"""
Stage 6a — Export our from-scratch checkpoint to Hugging Face format.

WHY:
    We built our own GPT class, but the world speaks "transformers". If we
    convert our weights into a standard `GPT2LMHeadModel`, then ANYONE can load
    our model with three lines:

        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("your-name/gpt-nano")
        tok   = AutoTokenizer.from_pretrained("your-name/gpt-nano")

    and it also becomes shareable on the Hugging Face Hub (stage 6b).

HOW:
    Our architecture is deliberately GPT-2 compatible. The only real difference
    is that we use nn.Linear where HF's GPT-2 uses Conv1D (which stores the
    weight transposed), so we transpose those weight matrices during copy.

Run:
    python stage6_publish/export_to_hf.py --ckpt checkpoints/tools.pt --out hf_model
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.chat_template import (  # noqa: E402
    ASSISTANT, BOS, EOS, PAD, SPECIAL_TOKENS, SYSTEM, TOOL_CALL, TOOL_RESULT, USER,
)
from common.model import GPT, GPTConfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
TOKENIZER_JSON = os.path.join(ROOT, "stage1_tokenizer", "tokenizer", "tokenizer.json")


def convert(ckpt_path: str, out_dir: str):
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    # 1) Load our checkpoint and rebuild our model.
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = GPTConfig(**ckpt["model_args"])
    ours = GPT(cfg)
    ours.load_state_dict(ckpt["model"])
    ours.eval()
    sd = ours.state_dict()

    # 2) Build an equivalent HF GPT-2 config.
    hf_config = GPT2Config(
        vocab_size=cfg.vocab_size,
        n_positions=cfg.block_size,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        activation_function="gelu",  # exact GELU, matches our nn.GELU()
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,  # no dropout at inference
        bos_token_id=None, eos_token_id=None,  # set below from the tokenizer
    )
    hf = GPT2LMHeadModel(hf_config)
    hf_sd = hf.state_dict()

    # 3) Map our weights onto HF's. These weights must be TRANSPOSED because HF
    #    GPT-2 uses Conv1D (weight shape [in, out]) vs our nn.Linear ([out, in]).
    transpose_suffixes = (
        "attn.c_attn.weight", "attn.c_proj.weight",
        "mlp.c_fc.weight", "mlp.c_proj.weight",
    )

    new_sd = {}
    # Embeddings
    new_sd["transformer.wte.weight"] = sd["transformer.wte.weight"]
    new_sd["transformer.wpe.weight"] = sd["transformer.wpe.weight"]
    # Final layer norm
    new_sd["transformer.ln_f.weight"] = sd["transformer.ln_f.weight"]
    new_sd["transformer.ln_f.bias"] = sd["transformer.ln_f.bias"]

    for i in range(cfg.n_layer):
        p = f"transformer.h.{i}."
        # layer norms and biases copy directly
        for name in ["ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias",
                     "attn.c_attn.bias", "attn.c_proj.bias",
                     "mlp.c_fc.bias", "mlp.c_proj.bias"]:
            new_sd[p + name] = sd[p + name]
        # weight matrices: transpose Linear -> Conv1D
        for name in ["attn.c_attn.weight", "attn.c_proj.weight",
                     "mlp.c_fc.weight", "mlp.c_proj.weight"]:
            new_sd[p + name] = sd[p + name].t().contiguous()

    # lm_head is tied to wte in both models; set it explicitly to be safe.
    new_sd["lm_head.weight"] = sd["lm_head.weight"]

    # 4) Load into the HF model (strict check on the keys we care about).
    missing, unexpected = hf.load_state_dict(new_sd, strict=False)
    # GPT-2 has some buffers (attn bias masks) that are not weights — those are
    # the only acceptable "missing" keys.
    real_missing = [m for m in missing if not (m.endswith(".attn.bias") or m.endswith(".attn.masked_bias"))]
    if real_missing:
        raise RuntimeError(f"Unmapped HF params: {real_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected params: {unexpected}")

    # 5) Wrap the trained tokenizer as a HF fast tokenizer with our special tokens.
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_JSON,
        bos_token=BOS, eos_token=EOS, pad_token=PAD, unk_token=None,
        additional_special_tokens=[SYSTEM, USER, ASSISTANT, TOOL_CALL, TOOL_RESULT],
    )
    hf.config.bos_token_id = tokenizer.bos_token_id
    hf.config.eos_token_id = tokenizer.eos_token_id
    hf.config.pad_token_id = tokenizer.pad_token_id

    # 6) Save everything.
    os.makedirs(out_dir, exist_ok=True)
    hf.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # 7) Also save the chat template so `apply_chat_template` works for users.
    _write_chat_template(tokenizer, out_dir)

    print(f"[export] saved HF model + tokenizer -> {out_dir}")
    return out_dir, hf, tokenizer


def _write_chat_template(tokenizer, out_dir):
    """Write a Jinja chat template matching common/chat_template.py so that
    `tokenizer.apply_chat_template(...)` produces the same strings we trained on."""
    template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}{{ '<|system|>' + message['content'] + '<|end|>' }}"
        "{% elif message['role'] == 'user' %}{{ '<|user|>' + message['content'] + '<|end|>' }}"
        "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>' + message['content'] + '<|end|>' }}"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
    )
    tokenizer.chat_template = template
    tokenizer.save_pretrained(out_dir)


def verify(out_dir: str):
    """Reload with plain transformers and generate, to prove the export works."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n[verify] reloading with transformers ...")
    model = AutoModelForCausalLM.from_pretrained(out_dir)
    tok = AutoTokenizer.from_pretrained(out_dir)
    model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful, concise assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tok(prompt, return_tensors="pt")
    out = model.generate(
        **inputs, max_new_tokens=40, do_sample=True, temperature=0.7, top_k=40,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=False)
    print("[verify] prompt+generation:")
    print(text)
    print("\n[verify] OK — the model loads and generates via transformers.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=os.path.join(ROOT, "checkpoints", "chat.pt"))
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "hf_model"))
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    out_dir, _, _ = convert(args.ckpt, args.out)
    if not args.no_verify:
        verify(out_dir)
    print("\nNext: python stage6_publish/push_to_hub.py --repo <your-username>/gpt-nano")


if __name__ == "__main__":
    main()
