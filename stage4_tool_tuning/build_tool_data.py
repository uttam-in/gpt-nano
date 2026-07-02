"""
Stage 4a — Build a dataset that teaches the model to CALL TOOLS.

The pattern we teach (one conversation):

    system      : <tools system prompt describing calculator/get_time/get_weather>
    user        : "What is 37 times 12?"
    tool_call   : {"name": "calculator", "arguments": {"expression": "37 * 12"}}
    tool_result : 444
    assistant   : "37 times 12 is 444."

Both the tool_call and the final assistant turn are TRAINED (see
common/chat_template.render_for_training). The tool_result is provided by the
environment at runtime, so it is NOT trained.

At inference (stage 5) the loop is:
    model emits <|tool_call|>{json}<|end|>  ->  we parse + run the tool
    ->  we feed <|tool_result|>...<|end|>   ->  model emits the final answer.

Output: stage4_tool_tuning/tool_data.jsonl

Run:
    python stage4_tool_tuning/build_tool_data.py --n 4000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stage4_tool_tuning.tools import (  # noqa: E402
    get_weather, run_tool, tools_system_prompt,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tool_data.jsonl")

CITIES = ["London", "Paris", "Tokyo", "New York", "Berlin", "Cairo"]


def tool_call_content(name: str, arguments: dict) -> str:
    # Compact JSON, stable key order, so the model sees a consistent format.
    return json.dumps({"name": name, "arguments": arguments}, separators=(", ", ": "))


def calc_convo(rng: random.Random) -> list[dict]:
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    op = rng.choice(["*", "+", "-"])
    expr = f"{a} {op} {b}"
    word = {"*": "times", "+": "plus", "-": "minus"}[op]
    q = rng.choice([f"What is {a} {word} {b}?", f"Compute {expr}.", f"Calculate {a} {word} {b}."])
    result = run_tool("calculator", {"expression": expr})
    return [
        {"role": "user", "content": q},
        {"role": "tool_call", "content": tool_call_content("calculator", {"expression": expr})},
        {"role": "tool_result", "content": result},
        {"role": "assistant", "content": f"{a} {word} {b} is {result}."},
    ]


def weather_convo(rng: random.Random) -> list[dict]:
    city = rng.choice(CITIES)
    q = rng.choice([f"What's the weather in {city}?", f"How is the weather in {city} today?"])
    result = get_weather(city)
    return [
        {"role": "user", "content": q},
        {"role": "tool_call", "content": tool_call_content("get_weather", {"city": city})},
        {"role": "tool_result", "content": result},
        {"role": "assistant", "content": f"The weather in {city} is {result}."},
    ]


def time_convo(rng: random.Random) -> list[dict]:
    q = rng.choice(["What time is it?", "Tell me the current time.", "What's the time now?"])
    return [
        {"role": "user", "content": q},
        {"role": "tool_call", "content": tool_call_content("get_time", {"timezone": "UTC"})},
        {"role": "tool_result", "content": "12:00 UTC"},
        {"role": "assistant", "content": "The current time is 12:00 UTC."},
    ]


def build(n: int, seed: int) -> list[list[dict]]:
    rng = random.Random(seed)
    system = {"role": "system", "content": tools_system_prompt()}
    generators = [calc_convo, weather_convo, time_convo]
    convos = []
    for _ in range(n):
        turns = rng.choices(generators, weights=[5, 3, 2])[0](rng)
        convos.append([system] + turns)
    return convos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    convos = build(args.n, args.seed)
    with open(OUT, "w", encoding="utf-8") as f:
        for c in convos:
            f.write(json.dumps({"messages": c}) + "\n")
    print(f"[done] wrote {len(convos):,} tool conversations -> {OUT}")
    print("\nExample:")
    print(json.dumps(convos[0], indent=2))
    print("\nNext: python stage4_tool_tuning/train_tools.py")


if __name__ == "__main__":
    main()
