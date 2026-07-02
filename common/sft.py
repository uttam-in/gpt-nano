r"""
Shared Supervised Fine-Tuning (SFT) machinery, used by stage 3 (chat) and
stage 4 (tool calling).

THE CORE IDEA — LOSS MASKING:
    When we fine-tune on a conversation, we do NOT want the model to be trained
    to predict the user's messages or the system prompt — only its own replies.
    So we build a `labels` tensor that equals the input IDs on the assistant's
    tokens and is -1 (ignored by the loss) everywhere else.

    Example (schematically):
        tokens : <|user|> what is 2+2 <|end|> <|assistant|> 4 <|end|>
        labels :   -1      -1  -1 -1    -1        4  <|end|>
                   \___________ ignored ________/  \__ trained __/

This file turns a list of conversations into padded (input_ids, labels) batches
ready for the same GPT model we pretrained.
"""

from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.chat_template import render_for_training  # noqa: E402
from common.tokenizer import ChatTokenizer  # noqa: E402

IGNORE_INDEX = -1  # must match ignore_index in common/model.py's cross_entropy


def encode_conversation(tok: ChatTokenizer, messages: list[dict], block_size: int):
    """Return (input_ids, labels) lists for ONE conversation.

    We render the conversation to a string plus the character spans the model
    should learn to produce, then map those spans onto token positions using the
    tokenizer's offset information.
    """
    text, spans = render_for_training(messages)

    encoding = tok.tok.encode(text)
    ids = encoding.ids
    offsets = encoding.offsets  # (start_char, end_char) for each token

    labels = [IGNORE_INDEX] * len(ids)
    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= tok_start:
            continue  # special tokens report a zero-width offset; skip them here
        # A token is "trainable" if its characters fall inside any assistant span.
        for span_start, span_end in spans:
            if tok_start >= span_start and tok_end <= span_end:
                labels[i] = ids[i]
                break

    # The special tokens (like <|end|>) inside a trainable span have zero-width
    # offsets, so also mark any special token whose char position sits in a span.
    # We detect them by re-checking their start offset against the spans.
    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_end == tok_start:  # special / zero-width token
            for span_start, span_end in spans:
                if span_start <= tok_start <= span_end:
                    labels[i] = ids[i]
                    break

    # Truncate to block_size. We keep the TAIL, not the front: the tokens we
    # actually train on (the assistant / tool_call turns) live at the END of the
    # conversation, so if it's too long we must preserve the end or we'd be left
    # with nothing trainable. (With the default block_size=256 our examples fit
    # and this never triggers; it matters for very small block sizes.)
    if len(ids) > block_size:
        ids = ids[-block_size:]
        labels = labels[-block_size:]
    return ids, labels


class ConversationDataset(Dataset):
    """A torch Dataset of tokenized, loss-masked conversations."""

    def __init__(self, conversations: list[list[dict]], tok: ChatTokenizer, block_size: int):
        self.examples = []
        skipped = 0
        for messages in conversations:
            ids, labels = encode_conversation(tok, messages, block_size)
            if all(l == IGNORE_INDEX for l in labels):
                skipped += 1
                continue  # no trainable tokens (e.g. truncated away) — drop it
            self.examples.append((ids, labels))
        if skipped:
            print(f"[sft] skipped {skipped} conversations with no trainable tokens")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, labels = self.examples[i]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def make_collate_fn(pad_id: int):
    """Return a collate function that pads a batch to the longest example.

    Inputs are padded with pad_id; labels are padded with IGNORE_INDEX so the
    padding never contributes to the loss.
    """

    def collate(batch):
        max_len = max(len(ids) for ids, _ in batch)
        input_ids, labels = [], []
        for ids, lab in batch:
            pad = max_len - len(ids)
            input_ids.append(torch.cat([ids, torch.full((pad,), pad_id, dtype=torch.long)]))
            labels.append(torch.cat([lab, torch.full((pad,), IGNORE_INDEX, dtype=torch.long)]))
        return torch.stack(input_ids), torch.stack(labels)

    return collate
