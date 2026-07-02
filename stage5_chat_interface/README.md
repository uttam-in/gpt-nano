# Stage 5 — Chat interfaces

Talk to your model, in the terminal or in a browser, with or without tools.

## Terminal chat with tools

```bash
python stage5_chat_interface/chat_with_tools.py            # tools run locally
python stage5_chat_interface/chat_with_tools.py --use-mcp  # tools run via MCP server
```

Try: `What is 47 times 19?` or `What's the weather in Tokyo?`
Commands: `/reset`, `/exit`.

## Web UI (Gradio)

```bash
# plain chat model
python stage5_chat_interface/web_app.py --ckpt ../checkpoints/chat.pt

# tool-calling model (shows tool calls in the chat)
python stage5_chat_interface/web_app.py --ckpt ../checkpoints/tools.pt --tools

# share a temporary public link
python stage5_chat_interface/web_app.py --ckpt ../checkpoints/chat.pt --share
```

Open the printed URL (default http://127.0.0.1:7860).

## Files

- `chat_with_tools.py` — terminal chat that runs the tool loop
  (`common/tool_loop.py`); `--use-mcp` routes tools through the stage-4 MCP
  server.
- `web_app.py` — a Gradio chat window; streams replies, optional tools.

## Tuning generation

`--temperature` (lower = more focused), `--top-k` (limit candidate tokens),
`--max-new-tokens` (reply length cap). These control sampling, not the model.

📖 Background: [`docs/05_tools_and_mcp.md`](../docs/05_tools_and_mcp.md)

➡️ Next: [`stage6_publish/`](../stage6_publish/)
