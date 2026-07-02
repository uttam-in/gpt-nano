# 04 — Chat fine-tuning: from text model to assistant

Covers stage 3 (`stage3_chat_finetune/`). We take the base model from stage 2
and teach it to hold a conversation.

## The problem

The base model only *continues* text. If you type "What is the capital of
France?", it might continue with more questions, because that's what its
training text looked like. We want it to instead produce a helpful *answer* and
then stop.

## The solution: supervised fine-tuning (SFT) on conversations

We keep training the model (same next-token objective), but now on data that is
formatted as conversations, using our special tokens:

```
<|bos|><|system|>You are a helpful, concise assistant.<|end|>
<|user|>What is 5 + 7?<|end|>
<|assistant|>The answer is 12.<|end|>
```

By seeing thousands of these, the model learns the pattern: *after `<|user|>...
<|end|><|assistant|>`, produce a helpful reply and emit `<|end|>` to stop.*

This is what `common/chat_template.py` builds. The same template is used for
training and for inference, which is essential — the model must see at inference
exactly the format it was trained on.

## The crucial trick: loss masking

We do **not** want to train the model to predict the user's messages or the
system prompt — only its own replies. If we trained on everything, the model
would waste capacity learning to generate user questions.

So we build a `labels` array that equals the token IDs on the **assistant's**
tokens and is `-1` (ignored by the loss) everywhere else:

```
tokens : <|user|> what is 5 + 7 <|end|> <|assistant|> The answer is 12 <|end|>
labels :   -1      -1  ...  -1     -1        The answer is 12 <|end|>   (trained)
           \_____________ ignored ____________/\________ trained ________/
```

`common/sft.py` does this by mapping each assistant text span to the tokens that
fall inside it. Only those positions contribute to the loss. This is the single
most important idea in instruction fine-tuning.

## One-token shift

Remember the model predicts token *t+1* from tokens up to *t*. So in the
training loop we feed `input_ids[:, :-1]` and compare against `labels[:, 1:]` —
the labels shifted by one. You'll see this in `train_chat.py`.

## Our (synthetic) chat data

Because our model is tiny and was pretrained on Shakespeare + simple stories, it
can't become a knowledgeable assistant. So `build_chat_data.py` generates
conversations it can actually master:

- greetings and small talk,
- simple arithmetic,
- "repeat this word" echo tasks,
- a handful of fixed factual Q&A.

The goal is to demonstrate that the model **learns the chat format and behavior**.
For a real project you'd swap in a human instruction dataset (Dolly,
OpenAssistant, Alpaca, etc.). The code doesn't change — same list of
`{"role", "content"}` messages.

## Running it

```bash
python stage3_chat_finetune/build_chat_data.py --n 4000
python stage3_chat_finetune/train_chat.py --epochs 3
python stage3_chat_finetune/chat_cli.py
```

You'll now be able to type messages and get replies that follow the format.
Expect simple, sometimes wrong answers — but it will *behave like a chatbot*.

## Why fine-tuning is cheap

Pretraining taught the model language from scratch (millions of tokens). Chat
fine-tuning only teaches a *format and style* on top — a few thousand short
examples and a few passes ("epochs") are enough. This is why everyone downloads
a base model and fine-tunes, rather than pretraining from zero.

## Common issues

- **Model never stops / rambles**: it isn't emitting `<|end|>`. Train a bit more,
  or check that your `<|end|>` token is inside the trained span (it is, in our
  `render_for_training`).
- **Model repeats the question**: loss masking may be off, or too few epochs.
- **Gibberish**: the base model probably wasn't trained enough in stage 2.

➡️ Next: [`05_tools_and_mcp.md`](05_tools_and_mcp.md).
