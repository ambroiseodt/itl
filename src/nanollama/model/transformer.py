"""
Transformer model.

#### Notes
Comments abbreviations:
    B: batch size
    S: sequence length
    D: embedding dimension
    H: number of heads

#### License
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from .blocklm import BlockLanguageModel, BlockLanguageModelConfig, BlockModel
from .feedforward import FeedForward
from .norm import RMSNorm

# At the moment, flex attention break debugpy, hence this flag to not use it.
FLEX_ATTENTION = False

# ------------------------------------------------------------------------------
# Attention Layer
# ------------------------------------------------------------------------------


def build_attention_mask(type: str, **kwargs: dict[str, Any]) -> BlockMask:
    """
    Build attention mask.

    ### Parameters
    type: type of attention, either "causal" or "doc"
    kwargs: additional arguments for the mask

    ### Returns
    mask: attention mask
    """
    if type == "causal":
        assert "seq_len" in kwargs, "sequence length must be provided for causal mask"
        seq_len = kwargs["seq_len"]
        return create_block_mask(_causal_mask_sign, None, None, seq_len, seq_len)

    else:
        raise ValueError(f"Unknown attention mask type: {type}")


def _causal_mask_sign(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    return q_idx >= kv_idx


class SelfAttention(nn.Module):
    """
    Self-attention layer.

    ### Parameters
    seq_len: maximum sequence length
    emb_dim: embedding dimensionality of the input
    nb_heads: number of attention heads (should divide emb_dim)
    nb_kv_heads: number of key-value heads (should divide nb_heads)
    rope_theta: rotational positional encoding parameter
    """

    def __init__(
        self,
        seq_len: int,
        emb_dim: int,
        nb_heads: int,
        nb_kv_heads: int,
        rope_theta: float,
    ):
        super().__init__()

        # dimensions
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.nb_heads = nb_heads
        self.nb_kv_heads = nb_kv_heads
        self.head_dim = self.emb_dim // self.nb_heads
        self.heads_per_group = self.nb_heads // self.nb_kv_heads
        assert self.emb_dim % self.nb_heads == 0, "embedding dimension must be divisible by number of heads"
        assert self.nb_heads % self.nb_kv_heads == 0, "number of heads must be divisible by number of key-value heads"

        # matrices
        self.W_query = nn.Linear(self.emb_dim, self.head_dim * self.nb_heads, bias=False)
        self.W_key = nn.Linear(self.emb_dim, self.head_dim * self.nb_kv_heads, bias=False)
        self.W_val = nn.Linear(self.emb_dim, self.head_dim * self.nb_kv_heads, bias=False)
        self.W_out = nn.Linear(self.nb_heads * self.head_dim, self.emb_dim, bias=False)

        # rotational positional encoding
        self.theta = rope_theta
        self.register_buffer(
            "rope_modulator", self._get_rope_modulator(self.seq_len, self.head_dim, self.theta), persistent=False
        )
        self.rope_modulator: torch.Tensor

    def reset_parameters(self, init_std: float, factor: float) -> None:
        """Weight initialization"""
        # input
        in_std = init_std or (self.emb_dim ** (-0.5))
        for W_in in [self.W_query, self.W_key, self.W_val]:
            nn.init.trunc_normal_(
                W_in.weight,
                mean=0.0,
                std=in_std,
                a=-3 * in_std,
                b=3 * in_std,
            )

        # output
        out_std = init_std or (self.emb_dim ** (-0.5))
        out_std = out_std / factor
        nn.init.trunc_normal_(
            self.W_out.weight,
            mean=0.0,
            std=out_std,
            a=-3 * in_std,
            b=3 * in_std,
        )

    def forward(self, x: torch.Tensor, kv_cache: Any = None, mask: BlockMask = None) -> torch.Tensor:
        """Self attention"""
        # dimensions
        bsz, seq_len, _ = x.size()

        # Query, key, value: (B, S, D) @ (D, D) -> (B, S, D)
        q, k, v = self.W_query(x), self.W_key(x), self.W_val(x)

        # reformating: (B, S, D) -> (B, S, H, D / H) -> (B, H, S, D / H)
        q, k, v = map(lambda t: t.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2), (q, k, v))

        # rope formatting
        q, k = map(lambda t: self._rope_view(t), (q, k))

        # KV cache for faster inference
        if kv_cache:
            k, v = kv_cache.update(k, v)

        # Flash attention implementation
        # ... -> (B, H, S, D / H)
        if FLEX_ATTENTION:
            z = flex_attention(q, k, v, block_mask=mask, enable_gqa=True)
        else:
            k, v = map(lambda t: torch.repeat_interleave(t, dim=2, repeats=self.heads_per_group), (k, v))
            z = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # reformating: (B, H, S, D / H) -> (B, S, H, D / H) -> (B, S, D)
        z = z.transpose(1, 2).reshape(bsz, seq_len, -1)

        # output layer: (B, L, D) @ (D, D) -> (N, L, D)
        z = self.W_out(z)
        return z

    @staticmethod
    def _get_rope_modulator(seq_len: int, dim: int, theta: float) -> torch.Tensor:
        """
        Returns the rope modulator for the attention mechanism.

        ### Parameters
        seq_len: sequence length
        dim: embedding dimension
        theta: rope angle parameter

        ### Returns
        rope_modulator: tensor of shape (seq_len, dim, 2) whose (t, k) element is
            .. math::
                \\cos(\\frac{2 \\pi t}{\\theta^{(2k / d)}}),
                \\sin(\\frac{2 \\pi t}{\\theta^{(2k / d)}})
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim - 1, 2) / dim))
        t = torch.arange(seq_len) * 2 * math.pi
        angles = torch.outer(t, freqs)
        cos, sin = angles.cos(), angles.sin()
        return torch.stack((cos, sin), dim=-1)

    def _rope_view(self, qk: torch.Tensor) -> torch.Tensor:
        """Recast tensor to complex numbers and apply rotational position filter."""
        B, H, S, dim = qk.size()
        assert S <= self.rope_modulator.size(0), "sequence length is too long for rope attention"

        rm = self.rope_modulator[:S].view(1, 1, S, dim // 2, 2)
        qk = qk.reshape(B, H, S, dim // 2, 2)

        # (x1 * cos - x2 * sin, x2 * cos + x1 * sin)
        # out = ((qk[..., 0] + qk[..., 1] * 1j) * (rm[..., 0] + rm[..., 1] * 1j))
        # out = torch.view_as_real(out)
        out = (qk[..., 0] * rm[..., 0] - qk[..., 1] * rm[..., 1], qk[..., 0] * rm[..., 1] + qk[..., 1] * rm[..., 0])
        out = torch.stack((out[0], out[1]), dim=-1)

        return out.type_as(qk).view((B, H, S, dim))


# ------------------------------------------------------------------------------
# Configuration Class
# ------------------------------------------------------------------------------


@dataclass
class TransformerBlockConfig:
    seq_len: int = 0
    emb_dim: int = 0

    # Transformer block parameters
    nb_heads: int = 0
    nb_kv_heads: int = 0
    rope_theta: float = 10_000
    hidden_dim: int = 0
    norm_eps: float = 1e-5

    def __check_init__(self):
        """Check validity of arguments that may have been inherited."""
        assert self.seq_len, "sequence length should be specified"
        assert self.emb_dim, "embedding dimension should be specified"
        assert self.hidden_dim, "hidden dimension should be specified"
        assert self.nb_heads, "number of heads should be specified"
        assert self.emb_dim // (2 * self.nb_heads), "embedding dimension should be divisible by 2 * number of heads"


# ------------------------------------------------------------------------------
# Transformer Block
# ------------------------------------------------------------------------------


class TransformerBlock(BlockModel):
    """
    Transformer block.

    ### Parameters
    config: configuration class containing arguments for SelfAttention and FeedForward
    """

    def __init__(self, config: TransformerBlockConfig):
        super().__init__()

        self.attn = SelfAttention(
            seq_len=config.seq_len,
            emb_dim=config.emb_dim,
            nb_heads=config.nb_heads,
            nb_kv_heads=config.nb_kv_heads,
            rope_theta=config.rope_theta,
        )
        self.ffn = FeedForward(emb_dim=config.emb_dim, hidden_dim=config.hidden_dim)
        self.attn_norm = RMSNorm(config.emb_dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.emb_dim, eps=config.norm_eps)

    def reset_parameters(self, init_std: float, factor: float) -> None:
        """Weight initialization"""
        self.attn.reset_parameters(init_std, factor)
        self.attn_norm.reset_parameters()
        self.ffn.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()

    def forward(self, x: torch.Tensor, mask: BlockMask = None) -> torch.Tensor:
        out = x + self.attn(self.attn_norm(x), mask=mask)
        out = out + self.ffn(self.ffn_norm(out))
        return out

    def get_nb_flop(self, mode: str = "both", seq_len: int = None) -> int:
        """
        TODO
        Number of flop to process a new token

        ### Parameters
        mode: whether to consider the forward, backward pass or both
        seq_len: sequence length
        """
        mode_multiplier = dict(fwd=1, bwd=2.5, both=3.5)[mode]
        return 0 * mode_multiplier


# ------------------------------------------------------------------------------
# Transformer Architecture
# ------------------------------------------------------------------------------


@dataclass
class TransformerConfig(BlockLanguageModelConfig):
    block: TransformerBlockConfig = field(default_factory=TransformerBlockConfig)

    def __post_init__(self):
        super().__post_init__()

        # Inherit parameters from the block model configuration.
        for attr in ["emb_dim", "norm_eps"]:
            setattr(self.block, attr, getattr(self, attr))

        # default scaling of ffn dimension
        if not self.block.hidden_dim:
            self.block.hidden_dim = 4 * self.emb_dim

        # no group query by default
        if not self.block.nb_kv_heads:
            self.block.nb_kv_heads = self.block.nb_heads


class Transformer(BlockLanguageModel):
    """Decoder only transformer."""

    def __init__(self, config: TransformerConfig):
        super().__init__(config, block=TransformerBlock)
        seq_len = config.block.seq_len
        self.mask = build_attention_mask("causal", seq_len=seq_len)

    def forward(self, x: torch.Tensor, mask: BlockMask = None) -> torch.Tensor:
        if mask is None:
            mask = self.mask
        out = self.embeddings(x)
        for layer in self.layers:
            out = layer(out, mask=mask)
        logits = self.output(self.output_norm(out))
        return logits
