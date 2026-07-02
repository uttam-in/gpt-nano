# 01 — Tokenization: turning text into numbers

Neural networks only do math on numbers. So the very first thing we need is a
way to convert text ↔ numbers. That's the **tokenizer**.

## Tokens, not characters or words

You might guess we'd feed the model one character at a time, or one word at a
time. In practice we use something in between: **subword tokens**.

```
"tokenization"  ->  ["token", "ization"]   (2 tokens)
"the"           ->  ["the"]                 (1 token)
"antidisestablishmentarianism" -> ["anti", "dis", "establish", "ment", ...]
```

Why subwords?
- **Characters** make sequences very long (slow, and the model wastes effort
  relearning how to spell common words).
- **Whole words** need a giant vocabulary and break on any word not seen in
  training.
- **Subwords** are a sweet spot: common words get one token, rare words split
  into reusable pieces, and *nothing is ever unrepresentable*.

## Byte-Pair Encoding (BPE)

We use **BPE**, the same algorithm family as GPT-2/3 and many others. The
training procedure is simple:

1. Start with the raw bytes as the initial "alphabet".
2. Count all adjacent pairs of tokens in the corpus.
3. Merge the **most frequent pair** into a single new token.
4. Repeat until you reach the target vocabulary size (we use 8192).

So if "t" followed by "h" is very common, BPE creates a "th" token; later "th"+
"e" might merge into "the". The learned merge rules ARE the tokenizer.

Because it starts from bytes, byte-level BPE can encode literally any text —
emoji, code, other languages — without an "unknown token".

## Special tokens

Beyond text, we reserve a few **special tokens** that carry structure. This
project uses (see `common/chat_template.py`):

| Token | Meaning |
|-------|---------|
| `<\|bos\|>` | beginning of a document/conversation |
| `<\|end\|>` | end of a turn or segment |
| `<\|system\|>` | start of system instructions |
| `<\|user\|>` | start of a user message |
| `<\|assistant\|>` | start of the model's reply |
| `<\|tool_call\|>` | the model is requesting a tool |
| `<\|tool_result\|>` | a tool's output being fed back |
| `<\|pad\|>` | filler to make batched sequences equal length |

These are single tokens the model can emit and recognize, which is how it learns
"a user turn just ended, now it's my turn to speak."

## What we build in stage 1

`stage1_tokenizer/train_tokenizer.py`:
- reads all `data/*.txt`,
- trains a byte-level BPE tokenizer with our special tokens reserved,
- saves it as `tokenizer.json` in **Hugging Face's format**, so later
  `transformers` and the Hub can load it directly.

Run it and watch the demo output — you'll see a sentence turn into token
strings and IDs, then decode back exactly.

## Encoding & decoding

```
encode:  "Hello world"  ->  [1529, 1128]
decode:  [1529, 1128]    ->  "Hello world"
```

`common/tokenizer.py` wraps this with a small `ChatTokenizer` class that also
knows the IDs of the special tokens. Every later stage uses it.

## Why the tokenizer decides the vocabulary size

The model's input and output layers have one slot per token in the vocabulary.
So the tokenizer's vocab size (8192) becomes the model's `vocab_size`. Bigger
vocab = shorter sequences but a bigger model; smaller vocab = the opposite. 8192
is a good small-scale choice.

➡️ Next: [`02_architecture.md`](02_architecture.md) — what's inside the model.
