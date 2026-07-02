"""
Stage 4c — Expose our toy tools over MCP (Model Context Protocol).

WHAT IS MCP?
    MCP is an open standard (by Anthropic) that lets AI apps talk to external
    "tool servers" in a uniform way. Instead of hard-coding a calculator into
    your chatbot, the chatbot connects to an MCP server that ADVERTISES tools
    and RUNS them on request. Any MCP-compatible client (Claude Desktop, our
    stage-5 chat app, etc.) can then use those tools.

    This is the same mechanism real assistants use to read files, query
    databases, call APIs — here we keep it tiny with calculator/time/weather.

This file starts a stdio MCP server. It's launched automatically by the
stage-5 tool chat when you pass --use-mcp, or you can register it with any MCP
client.

Run standalone (it will wait for a client on stdin/stdout):
    python stage4_tool_tuning/mcp_server.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stage4_tool_tuning.tools import TOOLS, run_tool  # noqa: E402

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise SystemExit("The `mcp` package is required. Install with: pip install mcp")

# FastMCP is the easy, decorator-based way to build an MCP server.
mcp = FastMCP("gpt-nano-tools")


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '12 * (3 + 4)'."""
    return run_tool("calculator", {"expression": expression})


@mcp.tool()
def get_time(timezone: str = "UTC") -> str:
    """Get the current time in the given timezone."""
    return run_tool("get_time", {"timezone": timezone})


@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city, e.g. 'London'."""
    return run_tool("get_weather", {"city": city})


if __name__ == "__main__":
    print(f"[mcp] serving tools: {', '.join(TOOLS)} (stdio)", file=sys.stderr)
    mcp.run()  # defaults to stdio transport
