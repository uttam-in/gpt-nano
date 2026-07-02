# Stage 4 — Tool calling & MCP

Teach the chat model to **call tools** (calculator, time, weather) and expose
those tools over **MCP (Model Context Protocol)**.

## Steps

```bash
# 1. Build tool-calling conversations
python stage4_tool_tuning/build_tool_data.py --n 4000

# 2. Fine-tune the CHAT model for tool calling
python stage4_tool_tuning/train_tools.py --epochs 3

# 3. (optional) Run the MCP server standalone
python stage4_tool_tuning/mcp_server.py
```

Then chat with tools in stage 5:

```bash
python stage5_chat_interface/chat_with_tools.py            # local tools
python stage5_chat_interface/chat_with_tools.py --use-mcp  # tools via MCP
```

## The pattern taught

```
<|user|>What is 47 times 19?<|end|>
<|tool_call|>{"name":"calculator","arguments":{"expression":"47 * 19"}}<|end|>
<|tool_result|>893<|end|>
<|assistant|>47 times 19 is 893.<|end|>
```

The model learns to emit the `tool_call`; our code runs the tool and feeds back
the `tool_result`; the model writes the final answer. Only `tool_call` and
`assistant` turns are trained (the result comes from the environment).

## Files

- `tools.py` — the tool functions + registry + the tools system prompt. Shared
  by the data generator, the local tool loop, and the MCP server.
- `build_tool_data.py` — generate `tool_data.jsonl`.
- `train_tools.py` — fine-tune (starts from `chat.pt`), saves `tools.pt`.
- `mcp_server.py` — expose the tools over MCP with the official `mcp` SDK.

## MCP in one line

MCP lets any compatible client (Claude Desktop, IDEs, our chat) discover and run
tools from a standard server, decoupling *who has the model* from *who has the
tools*. `--use-mcp` routes tool calls through `mcp_server.py` instead of calling
the Python functions directly — the model doesn't change.

📖 Background: [`docs/05_tools_and_mcp.md`](../docs/05_tools_and_mcp.md)

➡️ Next: [`stage5_chat_interface/`](../stage5_chat_interface/)
