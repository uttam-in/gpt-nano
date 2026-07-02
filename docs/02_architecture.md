# 02 — Architecture: inside the Transformer

This walks through `common/model.py`. Read them side by side. The model is a
**decoder-only Transformer**, the same design as GPT-2/3/4 and Llama.

## The big picture

```
token IDs ─► [token embedding] + [position embedding]
          ─► Transformer block  ┐
          ─► Transformer block   │  (repeated N times)
          ─► ...                 ┘
          ─► [final LayerNorm]
          ─► [linear head] ─► a score for every token in the vocabulary (logits)
          ─► softmax ─► probabilities ─► pick the next token
```

Our defaults (`GPTConfig`): 6 blocks, 6 attention heads, embedding size 384,
context length 256, vocab 8192 → about **14 million parameters**.

## Step 1: Embeddings — giving tokens meaning

A token ID like `1529` is just an index; it has no meaning by itself. The
**token embedding** (`wte`) is a lookup table that maps each token ID to a
vector of 384 numbers. During training these vectors organize themselves so that
similar tokens end up with similar vectors.

Transformers process all positions at once and have no built-in sense of order,
so we add a **position embedding** (`wpe`): a learned vector for "position 0",
"position 1", etc. Adding them tells the model both *what* the token is and
*where* it sits.

## Step 2: The Transformer block

Each block has two sub-layers, each wrapped in a LayerNorm and a residual
connection:

```
x = x + attention(layernorm(x))
x = x + mlp(layernorm(x))
```

### Self-attention — tokens looking at each other

This is the key idea. For each token, attention lets it **look at earlier tokens
and pull in relevant information**. Concretely, each token produces three
vectors:

- **Query (Q)**: "what am I looking for?"
- **Key (K)**: "what do I offer?"
- **Value (V)**: "here's my content."

A token compares its Query to every other token's Key to get attention weights
(how much to focus on each), then takes a weighted sum of their Values. So when
processing "it" in "the cat drank milk because it was thirsty", attention can
learn to pull information from "cat".

**Causal** means each token may only attend to positions **at or before** itself
— never the future. This is what makes the model a left-to-right *generator*.
In code it's the `is_causal=True` flag to `scaled_dot_product_attention`.

**Multi-head** means we do this several times in parallel (6 heads), each with
its own smaller Q/K/V, so different heads can specialize (one tracks syntax,
another tracks subjects, etc.). Their outputs are concatenated and mixed.

### MLP — the per-token "thinking"

After attention moves information *between* tokens, the MLP (a 2-layer
feed-forward network with a GELU activation) processes each token *individually*,
expanding to 4× width and back. This is where a lot of the model's "knowledge"
capacity lives.

### Residual connections & LayerNorm — why deep networks train

- **Residual** (`x = x + sublayer(x)`): the "+ x" gives gradients a clean path
  back through many layers, so stacking blocks doesn't stall learning.
- **LayerNorm**: rescales each token's vector to a stable range before each
  sub-layer, keeping the numbers well-behaved during training.

## Step 3: The output head

After the final LayerNorm, a linear layer (`lm_head`) projects each token's
384-dim vector to `vocab_size` (8192) **logits** — one raw score per possible
next token. `softmax` turns logits into probabilities.

**Weight tying**: we reuse the token-embedding matrix as the output projection
(`wte.weight = lm_head.weight`). This saves parameters and typically improves
quality — a standard GPT-2 trick.

## From logits to text: sampling

At generation time (`GPT.generate`) we:
1. Run the model to get logits for the last position.
2. Divide by **temperature** (lower = more focused/greedy, higher = more random).
3. Optionally keep only the **top-k** most likely tokens.
4. Sample one token from the resulting probabilities.
5. Append it and repeat until we hit `<|end|>` or a length limit.

## Why this design maps cleanly to GPT-2

We deliberately mirror GPT-2's structure (same block layout, learned position
embeddings, weight tying, GELU). That's what lets stage 6 convert our weights
into a Hugging Face `GPT2LMHeadModel` with just a few weight transposes.

➡️ Next: [`03_pretraining.md`](03_pretraining.md) — training the model.
