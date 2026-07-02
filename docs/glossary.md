# Glossary

Plain-language definitions of every term used in this project.

**Autoregressive** — generating text one token at a time, where each new token
depends on all the tokens before it.

**Attention (self-attention)** — the mechanism that lets each token look at other
tokens and pull in relevant information. See `docs/02_architecture.md`.

**Backpropagation** — the algorithm that computes, for every parameter, how a
small change would affect the loss. PyTorch does it automatically when you call
`loss.backward()`.

**Batch** — a group of training examples processed together for efficiency.
**Batch size** is how many.

**bfloat16 / float32** — number formats. float32 is full precision (4 bytes);
bfloat16 is half the size and faster, used on NVIDIA GPUs for training.

**Block / Transformer block** — one attention + MLP unit; the model stacks
several (`n_layer` of them).

**block_size** — the context length: how many tokens the model can see at once.

**BPE (Byte-Pair Encoding)** — the algorithm that builds the tokenizer by
repeatedly merging the most frequent adjacent token pair.

**Checkpoint** — a saved file (`.pt`) containing all the model's parameters, so
you can reload it later.

**Context window** — same as block_size; the span of tokens the model attends to.

**Cross-entropy loss** — the number measuring how far the model's predicted
probabilities are from the true next tokens. Training minimizes it.

**CUDA** — NVIDIA's GPU compute platform. Used automatically on NVIDIA hardware.

**Embedding** — a learned vector representing a token (token embedding) or a
position (position embedding).

**Epoch** — one full pass over the training dataset. Fine-tuning often uses a few
epochs.

**Fine-tuning** — continuing to train an already-trained model on new data (e.g.
conversations) to specialize it. Cheap compared to pretraining.

**GELU** — the activation function used inside the MLP; a smooth nonlinearity.

**Gradient** — the direction/magnitude to nudge a parameter to reduce loss.

**Gradient accumulation** — summing gradients over several small batches before
one optimizer step, to simulate a large batch on limited memory.

**Gradient clipping** — capping the size of gradients to prevent unstable steps.

**Head (attention head)** — one of several parallel attention computations; each
can specialize.

**Hugging Face / the Hub** — a company and website hosting open models, datasets,
and the `transformers`/`tokenizers`/`datasets` libraries.

**Inference** — using a trained model to generate output (no learning).

**Learning rate** — how big a step the optimizer takes each update. Too high
diverges; too low is slow.

**LayerNorm** — normalizes a token's vector to a stable range; helps deep
networks train.

**Logits** — the raw, unnormalized scores the model outputs for each possible
next token, before softmax.

**Loss** — see cross-entropy loss. The single number training drives down.

**Loss masking** — during fine-tuning, ignoring (label = -1) the tokens we don't
want the model trained to produce (system/user turns), so only its own replies
count.

**MCP (Model Context Protocol)** — an open standard for connecting AI apps to
external tool servers. See `docs/05_tools_and_mcp.md`.

**Memmap (memory-mapping)** — reading a file directly from disk as if it were an
array, without loading it all into RAM.

**MLP (feed-forward network)** — the per-token processing sub-layer inside each
Transformer block.

**MPS (Metal Performance Shaders)** — Apple's GPU compute backend, used
automatically on M1/M2/M3 Macs.

**Optimizer (AdamW)** — the algorithm that updates parameters using gradients.
AdamW is the standard choice for Transformers.

**Overfitting** — when a model memorizes the training data instead of
generalizing; spotted when validation loss rises while training loss falls.

**Parameter / weight** — one of the model's learnable numbers. Our model has
~14 million.

**Position embedding** — a learned vector encoding a token's position in the
sequence.

**Pretraining** — the first, large-scale training on raw text that gives a model
general language ability.

**Residual connection** — the `x = x + sublayer(x)` pattern that lets gradients
flow through deep networks.

**Sampling** — choosing the next token from the model's probability distribution.
Controlled by **temperature** and **top-k**.

**SFT (Supervised Fine-Tuning)** — fine-tuning on labeled examples (here,
conversations). Stages 3 and 4.

**Softmax** — converts logits into a probability distribution (positive, sums
to 1).

**Special tokens** — reserved tokens with structural meaning (`<|user|>`,
`<|end|>`, etc.) rather than ordinary text.

**Temperature** — sampling knob: <1 makes output more focused/deterministic, >1
more random.

**Token** — a chunk of text (word or word-piece) the model reads/writes, as an
integer ID.

**Tokenizer** — the component that converts text ↔ token IDs.

**Tool calling** — the model emitting a structured request for an external tool,
whose result is fed back so it can answer. Stage 4.

**top-k** — sampling knob: only consider the k most likely next tokens.

**Transformer** — the neural network architecture behind GPT and most modern LLMs.

**Validation set** — held-out data used to measure generalization, never trained
on.

**Vocabulary (vocab_size)** — the set of all tokens the tokenizer/model knows
(8192 here).

**Weight tying** — sharing the token-embedding matrix with the output layer to
save parameters and improve quality.
