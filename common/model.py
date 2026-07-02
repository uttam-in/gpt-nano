"""
A small GPT (decoder-only Transformer), written from scratch in ~250 lines.

This is deliberately close to GPT-2 so that we can later export the trained
weights into Hugging Face's `GPT2LMHeadModel` and share them on the Hub.

Read this file top-to-bottom to understand a Transformer. The pieces are:

    Token IDs ──► Token embedding + Position embedding
              ──► N × TransformerBlock (Attention + MLP)
              ──► Final LayerNorm
              ──► Linear head ──► logits over the vocabulary

Everything here is standard PyTorch. Nothing is Mac- or NVIDIA-specific — the
same code runs on MPS, CUDA, or CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    """All the knobs that define the model's size and shape.

    The defaults below describe a ~15M-parameter model — small enough to train
    on a Mac M1 in minutes-to-hours, big enough to produce coherent text.

    To scale UP later on your A6000, increase n_layer / n_head / n_embd.
    """

    vocab_size: int = 8192      # number of distinct tokens (set by the tokenizer)
    block_size: int = 256       # context length: how many tokens the model sees at once
    n_layer: int = 6            # number of Transformer blocks stacked on top of each other
    n_head: int = 6             # number of attention heads per block
    n_embd: int = 384           # embedding dimension (the "width" of the model)
    dropout: float = 0.1        # regularization; helps small models not memorize
    bias: bool = True           # use bias terms in Linear/LayerNorm (GPT-2 does)


# ---------------------------------------------------------------------------
# Building block 1: Causal self-attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    "Self-attention" lets every token look at earlier tokens and decide which
    ones are relevant. "Causal" means a token can only look BACKWARD (at the
    past), never forward — that's what makes it a language *generator*.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        # One big Linear produces Query, Key, and Value for all heads at once.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection: mixes the heads' outputs back together.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Compute Q, K, V and split into heads.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # reshape (B, T, C) -> (B, n_head, T, head_dim) so each head is independent
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # scaled_dot_product_attention handles the causal mask + softmax + dropout.
        # It uses a fast fused kernel on CUDA and a correct fallback on MPS/CPU.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,  # <-- the "look backward only" rule
        )

        # Re-assemble all heads side-by-side: (B, n_head, T, hd) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ---------------------------------------------------------------------------
# Building block 2: MLP (feed-forward network)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """A small 2-layer network applied to each position independently.

    Attention moves information *between* tokens; the MLP does the "thinking"
    *within* each token. The hidden layer is 4x wider, as in GPT-2.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Building block 3: One Transformer block
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """Attention + MLP, each wrapped in a LayerNorm and a residual connection.

    The "x = x + sublayer(norm(x))" pattern (pre-norm residual) is what lets us
    stack many blocks and still train stably.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# The full model
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),   # position embeddings
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),   # final layer norm
        ))
        # Language-model head: projects embeddings back to vocabulary logits.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the input embedding and output head share weights.
        # This is a standard trick that saves parameters and improves quality.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Special scaled init for residual projections (GPT-2 paper, section 2.3).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx:     (B, T) tensor of token IDs.
        targets: (B, T) tensor of the NEXT token at each position (for training).

        Returns (logits, loss). loss is None at inference time.
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)      # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)      # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training: compute logits at every position and the cross-entropy loss.
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # positions labelled -1 don't contribute (used for masking)
            )
        else:
            # Inference: we only need logits for the LAST position to predict the next token.
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Create an AdamW optimizer with weight decay only on 2D+ weights.

        Biases and LayerNorm gains are 1D and should NOT be decayed — decaying
        them hurts training. This is standard practice from the GPT-2/nanoGPT setup.
        """
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in param_dict.values() if p.dim() >= 2]
        no_decay = [p for p in param_dict.values() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # 'fused' AdamW is a faster CUDA-only kernel; plain AdamW everywhere else.
        use_fused = device_type == "cuda"
        extra = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token=None):
        """Autoregressively sample new tokens.

        idx: (B, T) starting context of token IDs.
        Returns idx extended by up to max_new_tokens.

        temperature: <1.0 = more focused/greedy, >1.0 = more random.
        top_k:       if set, sample only from the k most likely tokens.
        eos_token:   if set, stop early once this token is generated (batch size 1).
        """
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens if it's grown too long.
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token is not None and idx_next.item() == eos_token:
                break
        return idx


if __name__ == "__main__":
    # Quick sanity check: build the model and print its size.
    cfg = GPTConfig()
    model = GPT(cfg)
    print(f"Model config    : {cfg}")
    print(f"Total params    : {model.num_params():,}")
    print(f"Non-emb params  : {model.num_params(non_embedding=True):,}")
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(x, x)
    print(f"Forward OK      : logits {tuple(logits.shape)}, loss {loss.item():.3f}")
