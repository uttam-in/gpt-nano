"""
Stage 5b — A web chat interface (Gradio).

Opens a browser chat window so you can talk to your model with a real UI
instead of the terminal. Streams the reply token-by-token.

You can point it at any checkpoint:
    python stage5_chat_interface/web_app.py                      # chat model
    python stage5_chat_interface/web_app.py --ckpt checkpoints/tools.pt --tools

With --tools, the model can call the stage-4 tools and you'll see the tool
calls appear in the chat.

Then open the printed URL (http://127.0.0.1:7860) in your browser.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.chat_template import DEFAULT_SYSTEM_PROMPT  # noqa: E402
from common.inference import ChatModel  # noqa: E402
from common.tool_loop import run_with_tools  # noqa: E402
from stage4_tool_tuning.tools import run_tool, tools_system_prompt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "..", "checkpoints")


def build_ui(chat: ChatModel, use_tools: bool, temperature: float, top_k: int, max_new_tokens: int):
    import gradio as gr

    system_content = tools_system_prompt() if use_tools else DEFAULT_SYSTEM_PROMPT

    def respond(message, history):
        # Rebuild the conversation from Gradio's history (list of {role,content}).
        messages = [{"role": "system", "content": system_content}]
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})

        if use_tools:
            answer, trace = run_with_tools(
                chat, messages, lambda n, a: run_tool(n, a),
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_k=top_k, verbose=False,
            )
            prefix = ""
            for t in trace:
                prefix += f"🔧 `{t['name']}({t['arguments']})` → `{t['result']}`\n\n"
            yield prefix + answer
        else:
            prev = ""
            for full in chat.stream_reply(
                messages, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k,
            ):
                yield full
                prev = full

    title = "gpt-nano chat" + (" (with tools)" if use_tools else "")
    demo = gr.ChatInterface(
        fn=respond,
        title=title,
        description="A small GPT trained from scratch. Answers are simple by design.",
        examples=(
            ["What is 12 times 8?", "What's the weather in Tokyo?", "Hello!"]
            if use_tools else
            ["Hello!", "What is 5 + 7?", "What is the capital of France?"]
        ),
    )
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=os.path.join(CKPT_DIR, "chat.pt"))
    ap.add_argument("--tools", action="store_true", help="enable tool calling")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = ap.parse_args()

    print("Loading model ...")
    chat = ChatModel(args.ckpt, args.device)
    demo = build_ui(chat, args.tools, args.temperature, args.top_k, args.max_new_tokens)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
