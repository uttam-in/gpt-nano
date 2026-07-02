"""
The agent loop: let the model call tools, then answer.

Given a fine-tuned tool model (stage 4), a single user turn actually involves
several model generations:

    1. Model reads the conversation and generates a <|tool_call|> with JSON.
    2. We parse the JSON, run the tool, and append a <|tool_result|>.
    3. Model reads the result and generates its final <|assistant|> answer.
    (If the model answers directly without a tool call, we just return that.)

`tool_runner` is a function (name, arguments_dict) -> result_string. By default
it runs our local Python tools, but stage 5 can pass one backed by the MCP
server instead — the loop doesn't care where the tool actually runs.
"""

from __future__ import annotations

import json

import torch
from torch.nn import functional as F

from common.chat_template import (
    ASSISTANT, EOS, TOOL_CALL, TOOL_RESULT, render_conversation,
)


def _generate_segment(chat_model, prompt_text, max_new_tokens, temperature, top_k):
    """Generate raw token IDs until <|end|>, returning the decoded text AND
    which special token opened this segment isn't needed — we just return text."""
    tok = chat_model.tok
    ids = tok.encode(prompt_text)
    idx = torch.tensor(ids, dtype=torch.long, device=chat_model.device)[None, ...]
    eos_id = chat_model.eos_id
    generated = []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -chat_model.config.block_size:]
        logits, _ = chat_model.model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        nid = next_id.item()
        if nid == eos_id:
            break
        generated.append(nid)
        idx = torch.cat([idx, next_id], dim=1)
    return tok.decode(generated, skip_special_tokens=True)


def run_with_tools(
    chat_model,
    messages,
    tool_runner,
    max_new_tokens=200,
    temperature=0.7,
    top_k=40,
    max_tool_calls=3,
    verbose=True,
):
    """Run one assistant turn that may involve tool calls.

    Returns (final_answer, trace) where `trace` is a list of dicts describing
    each tool call for display/debugging.

    We decide whether the model wants a tool by rendering the conversation with
    the <|tool_call|> prompt appended and seeing if it produces valid JSON. To
    keep it simple and deterministic, we explicitly prompt for a tool call first;
    if the JSON doesn't parse or names no known tool, we fall back to a plain
    assistant answer.
    """
    trace = []
    working = list(messages)

    for _ in range(max_tool_calls):
        # Ask the model whether/what to call by priming the <|tool_call|> role.
        prompt = render_conversation(working) + TOOL_CALL
        raw = _generate_segment(chat_model, prompt, max_new_tokens, temperature, top_k)
        raw = raw.strip()

        call = _try_parse_tool_call(raw)
        if call is None:
            break  # model didn't produce a usable tool call -> answer directly

        name, arguments = call["name"], call.get("arguments", {})
        if verbose:
            print(f"  [tool call] {name}({arguments})")
        result = tool_runner(name, arguments)
        if verbose:
            print(f"  [tool result] {result}")
        trace.append({"name": name, "arguments": arguments, "result": result})

        # Record the tool_call and tool_result so the next generation sees them.
        working.append({"role": "tool_call", "content": json.dumps({"name": name, "arguments": arguments}, separators=(", ", ": "))})
        working.append({"role": "tool_result", "content": result})

    # Now produce the final assistant answer.
    prompt = render_conversation(working, add_generation_prompt=True)
    answer = _generate_segment(chat_model, prompt, max_new_tokens, temperature, top_k).strip()
    return answer, trace


def _try_parse_tool_call(text: str):
    """Extract a {"name":..., "arguments":...} object from generated text.

    The model may add stray characters; we grab the first {...} block and try to
    JSON-parse it. Returns the dict, or None if it isn't a valid tool call.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    blob = text[start:end + 1]
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    return obj
