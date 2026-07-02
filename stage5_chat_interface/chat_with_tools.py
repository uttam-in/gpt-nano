"""
Stage 5a — Terminal chat that can USE TOOLS.

This ties stage 4 together: you chat, and when the model decides it needs a
tool it emits a tool call, we run the tool, feed the result back, and it gives a
final answer.

Two ways to run the tools:
  - default (local):  tools run as plain Python via stage4_tool_tuning/tools.py
  - --use-mcp:        tools run through the MCP server (stage4_tool_tuning/mcp_server.py)
                      launched as a subprocess and called over the protocol.

Run (after stage 4 training):
    python stage5_chat_interface/chat_with_tools.py
    python stage5_chat_interface/chat_with_tools.py --use-mcp

Try:  "What is 47 times 19?"   or   "What's the weather in Tokyo?"
Commands:  /reset  /exit
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.inference import ChatModel  # noqa: E402
from common.tool_loop import run_with_tools  # noqa: E402
from stage4_tool_tuning.tools import run_tool, tools_system_prompt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "..", "checkpoints")


def make_local_runner():
    """Tool runner that executes tools in-process."""
    return lambda name, args: run_tool(name, args)


def make_mcp_runner():
    """Tool runner backed by the MCP server subprocess.

    We spin up mcp_server.py over stdio and call its tools via the MCP client.
    Everything is synchronous here for simplicity (we drive the async client
    with asyncio.run per call — fine for an interactive CLI).
    """
    import asyncio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server = os.path.join(HERE, "..", "stage4_tool_tuning", "mcp_server.py")
    params = StdioServerParameters(command=sys.executable, args=[server])

    def runner(name, arguments):
        async def _call():
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments)
                    # result.content is a list of content blocks; join their text.
                    parts = []
                    for block in result.content:
                        parts.append(getattr(block, "text", str(block)))
                    return "".join(parts)
        return asyncio.run(_call())

    return runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=os.path.join(CKPT_DIR, "tools.pt"))
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--use-mcp", action="store_true", help="run tools via the MCP server")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    print("Loading model ...")
    chat = ChatModel(args.ckpt, args.device)
    tool_runner = make_mcp_runner() if args.use_mcp else make_local_runner()
    mode = "MCP server" if args.use_mcp else "local functions"
    print(f"Ready. Tools run via: {mode}. Type /exit to quit, /reset to clear.\n")

    system = {"role": "system", "content": tools_system_prompt()}
    messages = [system]
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            messages = [system]
            print("(history cleared)\n")
            continue

        messages.append({"role": "user", "content": user})
        answer, trace = run_with_tools(
            chat, messages, tool_runner,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            verbose=True,
        )
        print(f"bot> {answer}\n")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
