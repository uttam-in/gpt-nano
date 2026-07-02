"""
A thin wrapper around the trained Hugging Face `tokenizers` file.

Every stage after stage 1 loads the tokenizer through this helper so the path
and the special-token IDs are defined in exactly one place.
"""

from __future__ import annotations

import os

from tokenizers import Tokenizer

from common.chat_template import BOS, EOS, PAD

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKENIZER_PATH = os.path.join(HERE, "..", "stage1_tokenizer", "tokenizer", "tokenizer.json")


class ChatTokenizer:
    """Encode/decode text and expose the IDs of the special tokens we care about."""

    def __init__(self, path: str = DEFAULT_TOKENIZER_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Tokenizer not found at {path}. "
                "Run stage1_tokenizer/train_tokenizer.py first."
            )
        self.tok = Tokenizer.from_file(path)
        self.bos_id = self.tok.token_to_id(BOS)
        self.eos_id = self.tok.token_to_id(EOS)
        self.pad_id = self.tok.token_to_id(PAD)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int:
        return self.tok.token_to_id(token)
