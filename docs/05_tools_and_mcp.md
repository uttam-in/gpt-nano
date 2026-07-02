# 05 — Tool calling and MCP

Covers stage 4 (`stage4_tool_tuning/`) and the tool loop used in stage 5.

## Why tools?

A language model only knows what's baked into its weights, and it's bad at
things like exact arithmetic or looking up live data. **Tool calling** fixes
this: instead of guessing, the model outputs a structured request like "run the
calculator on `47 * 19`", an external program runs it, and the *real result* is
fed back so the model can answer correctly.

This is the foundation of modern "agents": models that can search the web, read
files, query databases, send emails — by calling tools.

## The conversation pattern we teach

We extend the chat format with two more special tokens:

```
<|system|>   ...description of available tools...          <|end|>
<|user|>     What is 47 times 19?                           <|end|>
<|tool_call|>{"name": "calculator",
              "arguments": {"expression": "47 * 19"}}       <|end|>
<|tool_result|>893                                          <|end|>
<|assistant|>47 times 19 is 893.                            <|end|>
```

- The model generates the `<|tool_call|>` with a JSON request.
- *Our code* (not the model) parses the JSON, runs the tool, and inserts the
  `<|tool_result|>`.
- The model then reads the result and writes the final answer.

During fine-tuning, both the `tool_call` and the final `assistant` turn are
**trained** (the model must learn to produce them); the `tool_result` is
**not** trained (it comes from the environment). This masking is handled by
`common/chat_template.py` and `common/sft.py`.

## Our toy tools

`stage4_tool_tuning/tools.py` defines three simple tools:

- `calculator(expression)` — safe arithmetic (no `eval`!).
- `get_time(timezone)` — current time.
- `get_weather(city)` — a mocked weather lookup.

`build_tool_data.py` generates conversations using these tools, and
`train_tools.py` fine-tunes the **chat** model (from stage 3) to produce tool
calls. Starting from the chat model means it already knows the conversation
format, so it only has to learn the tool-calling pattern on top.

## The tool loop at inference (`common/tool_loop.py`)

When you chat with the tool model:

1. We prompt the model with the conversation + `<|tool_call|>` and let it
   generate a JSON request.
2. If it produces valid JSON naming a known tool, we run the tool and append the
   result; otherwise we skip straight to answering.
3. We prompt with `<|assistant|>` and let it write the final answer.

`_try_parse_tool_call` is deliberately forgiving — small models emit slightly
messy JSON, so we extract the first `{...}` block and try to parse it.

## What is MCP?

**MCP (Model Context Protocol)** is an open standard (from Anthropic) for
connecting AI apps to tool servers in a uniform way. Instead of hard-coding
tools into every app, you run an **MCP server** that advertises tools and
executes them on request. Any MCP-compatible client — Claude Desktop, IDEs, our
stage-5 chat — can then discover and call those tools over a standard protocol.

Why it matters: it decouples *who has the tools* from *who has the model*. You
can write a tool server once and use it from many apps; or connect your app to
tool servers other people wrote.

### Our MCP server

`stage4_tool_tuning/mcp_server.py` exposes the same three tools over MCP using
the official `mcp` SDK (`FastMCP`). Each tool is just a decorated Python
function:

```python
@mcp.tool()
def calculator(expression: str) -> str:
    "Evaluate an arithmetic expression."
    return run_tool("calculator", {"expression": expression})
```

Run it standalone (it waits for a client over stdio):

```bash
python stage4_tool_tuning/mcp_server.py
```

### Using the MCP server from the chat

`stage5_chat_interface/chat_with_tools.py --use-mcp` launches the MCP server as
a subprocess and routes every tool call through it via the MCP client, instead
of calling the Python functions directly. The model and the tool loop don't
change at all — only *where the tool runs* changes. That's the whole point of
MCP.

```bash
# tools run in-process:
python stage5_chat_interface/chat_with_tools.py
# tools run via the MCP server:
python stage5_chat_interface/chat_with_tools.py --use-mcp
```

## Scaling this up

Real tool use adds: many tools with rich JSON schemas, multiple tool calls per
turn, error handling and retries, and connecting to *external* MCP servers
(filesystem, GitHub, databases). The mechanics you learned here — structured
requests, feed the result back, answer — are exactly the same at any scale.

➡️ Next: [`06_publishing.md`](06_publishing.md).
