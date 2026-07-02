"""
Shared chat inference: load a fine-tuned checkpoint and generate assistant
replies from a conversation. Used by the CLI (stage 3), the tool-calling loop
(stage 4/5) and the Gradio web app (stage 5).
"""

from __future__ import annotations

import os
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.chat_template import ASSISTANT, EOS, render_conversation  # noqa: E402
from common.device import get_device  # noqa: E402
from common.model import GPT, GPTConfig  # noqa: E402
from common.tokenizer import ChatTokenizer  # noqa: E402


class ChatModel:
    """A loaded model + tokenizer with a convenient reply() method."""

    def __init__(self, ckpt_path: str, device_str: str | None = None):
        self.device = get_device(device_str)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.config = GPTConfig(**ckpt["model_args"])
        self.model = GPT(self.config)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.tok = ChatTokenizer()
        # We stop generation when the model emits <|end|>.
        self.eos_id = self.tok.token_to_id(EOS)

    @torch.no_grad()
    def stream_reply(self, messages, max_new_tokens=200, temperature=0.8, top_k=40):
        """Yield the assistant's reply token-by-token (as decoded text pieces).

        `messages` is a list of {"role","content"} dicts. We append the
        generation prompt (<|assistant|>) and generate until <|end|>.
        """
        prompt = render_conversation(messages, add_generation_prompt=True)
        ids = self.tok.encode(prompt)
        idx = torch.tensor(ids, dtype=torch.long, device=self.device)[None, ...]

        generated: list[int] = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            nid = next_id.item()
            if nid == self.eos_id:
                break
            generated.append(nid)
            idx = torch.cat([idx, next_id], dim=1)
            # Decode incrementally so the caller can print as it goes.
            yield self.tok.decode(generated, skip_special_tokens=True)

    def reply(self, messages, **kwargs) -> str:
        """Return the full assistant reply as a string."""
        text = ""
        for text in self.stream_reply(messages, **kwargs):
            pass
        return text
