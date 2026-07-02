# 00 — Fundamentals: what is a language model?

*No prior machine-learning knowledge assumed. Read this first.*

## The one-sentence idea

A language model is a program that, given some text, predicts **what token comes
next**. That's it. Everything else — chatting, answering questions, calling
tools — is built on top of this single skill.

```
Input:  "The capital of France is"
Model:  predicts the next token is most likely " Paris"
```

Do this repeatedly — predict a token, add it to the text, predict again — and
the model *generates* new text. This is called **autoregressive generation**.

## Why "predict the next token" is enough

To predict the next token well across billions of examples, a model is forced to
learn a surprising amount:

- **Grammar**: after "The cat sat on the", "mat" is more likely than "purple".
- **Facts**: after "The capital of France is", "Paris" is likely.
- **Reasoning patterns**: after "2 + 2 =", "4" is likely.

Nobody programs these rules. The model discovers them by adjusting millions of
internal numbers ("parameters") to make its predictions match real text. That
adjustment process is **training**.

## The three phases of building a chat model

This project walks through all three:

1. **Pretraining** (stage 2): show the model a mountain of raw text and have it
   practice next-token prediction. This builds general language ability. It's
   the expensive part. Result: a "base model" that continues text but doesn't
   follow instructions.

2. **Fine-tuning for chat** (stage 3): continue training, but now on
   *conversations*, so the model learns to read a user's message and produce a
   helpful reply. This is cheap compared to pretraining. Result: an "instruct"
   or "chat" model.

3. **Fine-tuning for tools** (stage 4): teach the model to output a structured
   request ("call the calculator with 12*8") when it needs external help, then
   use the result. Result: a model that can *act*, not just talk.

## Key vocabulary (see `glossary.md` for more)

- **Token**: a chunk of text the model reads/writes — often a word or word-piece
  like "ing". Models work with token **IDs** (integers), not raw characters.
- **Parameter / weight**: one of the millions of numbers inside the model that
  get adjusted during training. Our model has ~14 million; GPT-3 had 175 billion.
- **Training**: repeatedly showing the model text and nudging its parameters so
  its predictions improve.
- **Loss**: a single number measuring how wrong the model's predictions are.
  Training tries to make loss go **down**. Lower loss = better predictions.
- **Inference**: using a trained model to generate text (no learning happens).
- **Checkpoint**: a saved snapshot of all the model's parameters (a `.pt` file
  here) that you can reload later.

## How training actually works (the 60-second version)

1. Take a chunk of text, e.g. tokens `[The, cat, sat]`.
2. Ask the model to predict the next token at every position.
3. Compare its predictions to the real next tokens (`[cat, sat, on]`) → compute
   the **loss**.
4. Use calculus (**backpropagation**) to find, for each parameter, which
   direction to nudge it to reduce the loss.
5. Nudge all parameters a tiny step in that direction (the **optimizer**).
6. Repeat millions of times with different chunks.

You don't need to implement the calculus — PyTorch does it automatically. You'll
watch the loss number drop as training proceeds, which is deeply satisfying.

## What "small" means here, and why we do it

Real models are trained on trillions of tokens across thousands of GPUs for
weeks. We use ~2.5 million tokens and one laptop GPU for minutes-to-hours. Our
model will produce simple, often silly text — but it goes through *exactly the
same steps* as the big ones. Learning the pipeline at small scale is the fastest
way to understand the whole field.

➡️ Next: [`01_tokenization.md`](01_tokenization.md) — how text becomes numbers.
