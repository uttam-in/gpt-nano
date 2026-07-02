"""
Stage 4 — The toy tools our model can call.

Each tool has:
  - a name
  - a short description + argument schema (shown to the model in the system prompt)
  - a Python function that actually runs it

These same tools are exposed two ways:
  - directly, for generating training data and for the local chat loop
  - over MCP (Model Context Protocol) in mcp_server.py

Keeping the definitions here means the training data and the real runtime use
IDENTICAL tools — so what the model learned matches what actually executes.
"""

from __future__ import annotations

import ast
import operator


# ---- calculator ----------------------------------------------------------
# A safe arithmetic evaluator (no eval()!). Supports + - * / and parentheses.
_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numbers allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '12 * (3 + 4)'."""
    result = _safe_eval(ast.parse(expression, mode="eval"))
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# ---- get_time ------------------------------------------------------------
def get_time(timezone: str = "UTC") -> str:
    """Return the current time. (Deterministic-ish; timezone is echoed.)"""
    import datetime
    now = datetime.datetime.utcnow().strftime("%H:%M")
    return f"{now} {timezone}"


# ---- weather (mocked) ----------------------------------------------------
_FAKE_WEATHER = {
    "london": "15°C and cloudy",
    "paris": "18°C and sunny",
    "tokyo": "22°C and rainy",
    "new york": "20°C and clear",
}


def get_weather(city: str) -> str:
    """Return a (fake) weather report for a city."""
    return _FAKE_WEATHER.get(city.strip().lower(), f"18°C and mild in {city}")


# ---- registry -------------------------------------------------------------
# name -> (function, description, {arg: description})
TOOLS = {
    "calculator": (
        calculator,
        "Evaluate an arithmetic expression.",
        {"expression": "a math expression, e.g. '12 * 8'"},
    ),
    "get_time": (
        get_time,
        "Get the current time in a timezone.",
        {"timezone": "timezone name, e.g. 'UTC'"},
    ),
    "get_weather": (
        get_weather,
        "Get the current weather for a city.",
        {"city": "city name, e.g. 'London'"},
    ),
}


def run_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with a dict of arguments. Returns a string result."""
    if name not in TOOLS:
        return f"error: unknown tool '{name}'"
    fn = TOOLS[name][0]
    try:
        return str(fn(**arguments))
    except Exception as e:  # noqa: BLE001 — surface tool errors back to the model
        return f"error: {e}"


def tools_system_prompt() -> str:
    """Build the system prompt that tells the model which tools exist and how to
    call them. This text must be present at inference time too (stage 5)."""
    lines = [
        "You are a helpful assistant that can call tools.",
        "When a tool is needed, respond with a tool call in this exact JSON form:",
        '{"name": "<tool_name>", "arguments": {...}}',
        "After you receive the tool result, give a short final answer.",
        "",
        "Available tools:",
    ]
    for name, (_, desc, schema) in TOOLS.items():
        args = ", ".join(f"{k} ({v})" for k, v in schema.items())
        lines.append(f"- {name}: {desc} arguments: {args}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(tools_system_prompt())
    print("\nself-test:")
    print("  calculator('12 * (3+4)') =", calculator("12 * (3+4)"))
    print("  get_weather('London')    =", get_weather("London"))
    print("  run_tool bad             =", run_tool("nope", {}))
