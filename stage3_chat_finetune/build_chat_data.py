"""
Stage 3a — Build a small chat (instruction-following) dataset.

Our base model is tiny (~15M params) and was pretrained on Shakespeare +
simple stories. It cannot become a knowledgeable assistant. But it CAN learn
the *shape* of a conversation: read a user turn, then produce an assistant turn
and stop. That is exactly what this dataset teaches.

We generate synthetic conversations that a small model can actually master:
  - greetings and small talk
  - simple arithmetic (addition, subtraction, multiplication)
  - "repeat after me" / echo tasks
  - a handful of fixed factual Q&A

For a REAL project you'd swap this for a human instruction dataset (e.g.
Databricks Dolly, OpenAssistant, or Alpaca). The format is identical — a list
of {"role", "content"} messages — so nothing downstream changes.

Output: stage3_chat_finetune/chat_data.jsonl  (one conversation per line)

Run:
    python stage3_chat_finetune/build_chat_data.py --n 4000
"""

from __future__ import annotations

import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "chat_data.jsonl")

SYSTEM = "You are a helpful, concise assistant."

GREETINGS = [
    ("Hi!", "Hello! How can I help you today?"),
    ("Hello", "Hi there! What can I do for you?"),
    ("Hey, how are you?", "I'm doing well, thanks for asking! How can I help?"),
    ("Good morning", "Good morning! How can I help you today?"),
    ("Thanks!", "You're welcome! Let me know if there's anything else."),
    ("Thank you so much", "You're very welcome!"),
    ("Bye", "Goodbye! Have a great day."),
    ("Who are you?", "I'm a small language model here to help answer your questions."),
    ("What can you do?", "I can chat, do simple arithmetic, and answer basic questions."),
]

FACTS = [
    ("What color is the sky on a clear day?", "The sky is blue on a clear day."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("How many legs does a spider have?", "A spider has eight legs."),
    ("What do bees make?", "Bees make honey."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
    ("How many months are in a year?", "There are twelve months in a year."),
    ("What sound does a dog make?", "A dog says woof."),
]


def arithmetic_example(rng: random.Random) -> list[dict]:
    op = rng.choice(["+", "-", "*"])
    a = rng.randint(0, 20)
    b = rng.randint(0, 20)
    if op == "+":
        ans = a + b
        q = f"What is {a} + {b}?"
    elif op == "-":
        ans = a - b
        q = f"What is {a} - {b}?"
    else:
        ans = a * b
        q = f"What is {a} times {b}?"
    templates = [
        f"{a} {op} {b} equals {ans}.",
        f"The answer is {ans}.",
        f"{ans}.",
    ]
    return [
        {"role": "user", "content": q},
        {"role": "assistant", "content": rng.choice(templates)},
    ]


def echo_example(rng: random.Random) -> list[dict]:
    words = ["apple", "river", "silver", "quiet", "dragon", "coffee", "mountain",
             "yellow", "gentle", "puzzle", "planet", "garden"]
    word = rng.choice(words)
    q = rng.choice([f"Repeat the word '{word}'.", f"Say '{word}'.", f"Can you say '{word}'?"])
    return [
        {"role": "user", "content": q},
        {"role": "assistant", "content": word},
    ]


def fixed_example(rng: random.Random, pool) -> list[dict]:
    q, a = rng.choice(pool)
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


def build(n: int, seed: int) -> list[list[dict]]:
    rng = random.Random(seed)
    convos = []
    generators = [
        lambda: fixed_example(rng, GREETINGS),
        lambda: fixed_example(rng, FACTS),
        lambda: arithmetic_example(rng),
        lambda: echo_example(rng),
    ]
    for _ in range(n):
        turns = rng.choices(generators, weights=[3, 3, 4, 2])[0]()
        convo = [{"role": "system", "content": SYSTEM}] + turns
        convos.append(convo)
    return convos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="number of conversations")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    convos = build(args.n, args.seed)
    with open(OUT, "w", encoding="utf-8") as f:
        for c in convos:
            f.write(json.dumps({"messages": c}) + "\n")

    print(f"[done] wrote {len(convos):,} conversations -> {OUT}")
    print("\nExample:")
    print(json.dumps(convos[0], indent=2))
    print("\nNext: python stage3_chat_finetune/train_chat.py")


if __name__ == "__main__":
    main()
