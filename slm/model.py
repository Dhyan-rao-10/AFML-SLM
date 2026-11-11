"""Decoder-only transformer with swappable components, built for ablations.

Each architecture choice the project tests is a config switch, so a GPT-2-style
baseline and a LLaMA-style model are the same class with different settings:

    pos_emb    "learned" (GPT-2)    | "rope" (LLaMA)
    norm       "layernorm" (GPT-2)  | "rmsnorm" (LLaMA)
    mlp        "gelu" (GPT-2)       | "swiglu" (LLaMA)
    n_kv_head  == n_head (multi-head attention) | < n_head (grouped-query attention)

All variants use pre-norm residual blocks, bias-free linear layers and tied
input/output embeddings, so only the switched component differs between runs.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int | None = None  # None -> n_head (standard multi-head attention)
    d_model: int = 384
    pos_emb: str = "learned"
    norm: str = "layernorm"
    mlp: str = "gelu"
    mlp_hidden: int | None = None  # None -> parameter-matched default, see __post_init__
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.d_model % self.n_head == 0, "d_model must divide evenly into heads"
        assert self.n_head % self.n_kv_head == 0, "n_head must be a multiple of n_kv_head"
        assert self.pos_emb in ("learned", "rope")
        assert self.norm in ("layernorm", "rmsnorm")
        assert self.mlp in ("gelu", "swiglu")
        if self.pos_emb == "rope":
            assert (self.d_model // self.n_head) % 2 == 0, "RoPE needs an even head dimension"
        if self.mlp_hidden is None:
            if self.mlp == "gelu":
                self.mlp_hidden = 4 * self.d_model
            else:
                # SwiGLU has three weight matrices instead of two. A hidden size of
                # 8/3 * d gives 3 * d * (8/3 * d) = 8 * d^2 weights, the same as GELU's
                # 2 * d * 4d. Round up to a multiple of 64 for GPU-friendly shapes.
                hidden = int(8 * self.d_model / 3)
                self.mlp_hidden = 64 * math.ceil(hidden / 64)


class RMSNorm(nn.Module):
    """Scale by the root-mean-square of the features. Unlike LayerNorm there is no
    mean subtraction and no bias, which saves compute and (empirically) loses little."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_f.type_as(x) * self.weight


def make_norm(cfg: GPTConfig) -> nn.Module:
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.d_model)
    return nn.LayerNorm(cfg.d_model)


def rope_cache(head_dim: int, max_len: int, theta: float):
    """Precompute the rotation angles for rotary position embeddings.

    Feature pair i rotates at frequency theta^(-2i/head_dim): early pairs spin fast
    (fine position), late pairs spin slowly (coarse position)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)  # (T, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate query/key features by their position. x: (B, H, T, head_dim).

    Splits the head into two halves and treats (x1[i], x2[i]) as a 2D point rotated
    by angle[t, i]. The dot product of a rotated query and key then depends only on
    their offset, which is what makes RoPE a relative position encoding."""
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.dropout = cfg.dropout
        # Separate projections (not one fused qkv matrix) so grouped-query attention
        # can shrink k/v and LoRA can target each projection by name.
        self.q_proj = nn.Linear(cfg.d_model, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        if self.n_kv_head != self.n_head:
            # Grouped-query attention: each k/v head is shared by a group of query heads.
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.o_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.kind = cfg.mlp
        if cfg.mlp == "swiglu":
            self.gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        if self.kind == "swiglu":
            # A learned, input-dependent gate decides how much of each hidden unit passes.
            h = F.silu(self.gate(x)) * self.up(x)
        else:
            h = F.gelu(self.up(x))
        return self.drop(self.down(h))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = make_norm(cfg)
        self.attn = Attention(cfg)
        self.norm2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, rope=None):
        # Pre-norm: normalize the input to each sublayer, keep the residual path clean.
        x = x + self.attn(self.norm1(x), rope)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model) if cfg.pos_emb == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        if cfg.pos_emb == "rope":
            cos, sin = rope_cache(cfg.d_model // cfg.n_head, cfg.block_size, cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 scaling: each block adds two residual branches, so shrink the output
        # projections by 1/sqrt(2 * n_layer) to keep the residual stream's variance flat.
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights are counted once
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. targets: (B, T) next-token ids, -100 = ignore.
        Without targets, returns logits for the last position only (for sampling)."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        rope = (self.rope_cos[:T], self.rope_sin[:T]) if self.cfg.pos_emb == "rope" else None
        for block in self.blocks:
            x = block(x, rope)
        x = self.norm_f(x)
        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
        )
        return logits, loss

    def configure_optimizer(self, lr, weight_decay, betas, device_type):
        # Decay matrices only; norm gains are 1D and should not be pulled toward zero.
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=device_type == "cuda")

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_ids=()):
        """Sample tokens autoregressively. Stops early at a stop id (batch size 1 only)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float()
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits[logits < kth] = -float("inf")
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_ids and idx.size(0) == 1 and next_id.item() in stop_ids:
                break
        return idx
