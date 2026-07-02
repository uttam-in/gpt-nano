"""
The chat template: how we turn a conversation into a flat string of tokens.

A base language model only knows how to continue text. To make it *chat*, we
agree on a fixed format that marks who is speaking. We use special tokens:

    <|system|>   ... system instructions ...   <|end|>
    <|user|>     ... the user's message ...     <|end|>
    <|assistant| >... the model's reply ...      <|end|>

During fine-tuning (stage 3) we train the model to produce the assistant part.
During chat (stage 5) we feed everything up to "<|assistant|>" and let the model
generate until it emits "<|end|>".

Tool calling (stage 4) reuses the same idea with two more tokens:

    <|tool_call|> {"name": "...", "arguments": {...}} <|end|>
    <|tool_result|> ...output from the tool... <|end|>

These strings must EXACTLY match the special tokens we register in the tokenizer
(stage 1), so they are defined here in one place and imported everywhere.
"""

from __future__ import annotations

# The special tokens. Order matters only in that the tokenizer must know all of them.
BOS = "<|bos|>"                 # beginning of a document/conversation
EOS = "<|end|>"                 # end of any turn or segment
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
TOOL_CALL = "<|tool_call|>"     # assistant is requesting a tool
TOOL_RESULT = "<|tool_result|>" # environment returns a tool's output
PAD = "<|pad|>"                 # padding for batching

SPECIAL_TOKENS = [BOS, EOS, SYSTEM, USER, ASSISTANT, TOOL_CALL, TOOL_RESULT, PAD]

DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant."


def render_conversation(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Turn a list of message dicts into the flat training/inference string.

    messages: [{"role": "system"|"user"|"assistant"|"tool_call"|"tool_result",
                "content": "..."}, ...]

    add_generation_prompt: if True, end the string with "<|assistant|>" so the
        model knows it's its turn to speak (used at inference time).
    """
    role_token = {
        "system": SYSTEM,
        "user": USER,
        "assistant": ASSISTANT,
        "tool_call": TOOL_CALL,
        "tool_result": TOOL_RESULT,
    }
    out = BOS
    for m in messages:
        tok = role_token[m["role"]]
        out += f"{tok}{m['content']}{EOS}"
    if add_generation_prompt:
        out += ASSISTANT
    return out


def render_for_training(messages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Like render_conversation, but also returns the character spans that the
    model should be TRAINED to produce (the assistant + tool_call replies).

    We return spans as (start, end) character offsets into the rendered string.
    Stage 3 converts these into a token-level loss mask so the model is only
    penalized for its own words, not for the user's or system's.
    """
    role_token = {
        "system": SYSTEM,
        "user": USER,
        "assistant": ASSISTANT,
        "tool_call": TOOL_CALL,
        "tool_result": TOOL_RESULT,
    }
    trainable_roles = {"assistant", "tool_call"}
    out = BOS
    spans: list[tuple[int, int]] = []
    for m in messages:
        tok = role_token[m["role"]]
        out += tok
        start = len(out)
        out += m["content"]
        out += EOS
        end = len(out)  # include the EOS token in the trainable span
        if m["role"] in trainable_roles:
            spans.append((start, end))
    return out, spans


if __name__ == "__main__":
    convo = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there, how can I help?"},
    ]
    print("--- inference string (add_generation_prompt=True) ---")
    print(repr(render_conversation(convo[:2], add_generation_prompt=True)))
    print("\n--- training string + trainable spans ---")
    s, spans = render_for_training(convo)
    print(repr(s))
    for a, b in spans:
        print(f"  trainable: {s[a:b]!r}")
