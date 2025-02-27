"""
Transformer model.

#### License
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

# Comments abbreviations:
#     B: batch size
#     S: sequence length
#     D: embedding dimension
#     H: number of heads

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

FLEX_ATTENTION = False


# ------------------------------------------------------------------------------
# Attention Mask
# ------------------------------------------------------------------------------


def build_attention_mask(type: str, **kwargs: dict[str, Any]) -> BlockMask:
    """
    Build attention mask.

    ### Parameters
    - type: type of attention, either "causal" or "doc"
    - kwargs: additional arguments for the mask

    ### Returns
    - mask: attention mask
    """
    if type == "causal":
        assert "seq_len" in kwargs, "sequence length must be provided for causal mask"
        seq_len = kwargs["seq_len"]
        return create_block_mask(_causal_mask_sign, None, None, seq_len, seq_len)

    else:
        raise ValueError(f"Unknown attention mask type: {type}")


def _causal_mask_sign(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    return q_idx >= kv_idx


# ------------------------------------------------------------------------------
# Attention Cache
# ------------------------------------------------------------------------------


@dataclass
class KVCache:
    """
    Key-Value cache for faster inference.

    ### Parameters
    - key: prefilled key tensor
    - value: prefilled value tensor
    - pos_idx: current position for each sentence in the cache
    """

    key: torch.Tensor
    value: torch.Tensor
    pos_idx: int = 0

    def __call__(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache with new key and value tensors.

        ### Parameters
        - key: key tensor
        - value: value tensor

        ### Returns
        - key: updated key tensor (concatenating past values with new ones)
        - value: updated value tensor
        """

        # Sequence length is the third dimension, (B, H, S, D / H)
        seq_len = key.size(2)
        self.key[..., self.pos_idx : self.pos_idx + seq_len, :] = key
        self.value[..., self.pos_idx : self.pos_idx + seq_len, :] = value
        self.pos_idx += seq_len
        return self.key[..., : self.pos_idx, :], self.value[..., : self.pos_idx, :]

    def reset(self) -> None:
        self.pos_idx = 0


# ------------------------------------------------------------------------------
# Attention Layer
# ------------------------------------------------------------------------------


class SelfAttention(nn.Module):
    """
    Self-attention layer.

    ### Parameters
    - seq_len: maximum sequence length
    - emb_dim: embedding dimensionality of the input
    - nb_heads: number of attention heads (should divide emb_dim)
    - nb_kv_heads: number of key-value heads (should divide nb_heads)
    - rope_theta: rotational positional encoding parameter
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
        self.head_dim = self.emb_dim // nb_heads
        self.heads_per_group = nb_heads // nb_kv_heads
        assert self.emb_dim % nb_heads == 0, "embedding dimension must be divisible by number of heads"
        assert nb_heads % nb_kv_heads == 0, "number of heads must be divisible by number of key-value heads"

        # matrices
        self.W_query = nn.Linear(self.emb_dim, self.head_dim * nb_heads, bias=False)
        self.W_key = nn.Linear(self.emb_dim, self.head_dim * nb_kv_heads, bias=False)
        self.W_val = nn.Linear(self.emb_dim, self.head_dim * nb_kv_heads, bias=False)
        self.W_out = nn.Linear(nb_heads * self.head_dim, self.emb_dim, bias=False)

        # rotational positional encoding
        self.theta = rope_theta
        self.register_buffer(
            "rope_modulator", self._get_rope_modulator(self.seq_len, self.head_dim, self.theta), persistent=False
        )
        self.rope_modulator: torch.Tensor

    def reset_parameters(self, init_std: float, factor: float) -> None:
        """
        Resetting module parameters

        ### Parameters
        - init_std: standard deviation of the initialization
        - factor: scaling factor for the output layer
        """
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

    def forward(self, x: torch.Tensor, kv_cache: KVCache = None, mask: BlockMask = None) -> torch.Tensor:
        """Self attention"""
        # dimensions
        bsz, seq_len, _ = x.size()

        # Query, key, value: (B, S, D) @ (D, D) -> (B, S, D)
        q, k, v = self.W_query(x), self.W_key(x), self.W_val(x)

        # reformating: (B, S, D) -> (B, S, H, D / H) -> (B, H, S, D / H)
        q, k, v = map(lambda t: t.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2), (q, k, v))

        # rope formatting
        # ... retrieve position in sequence if generating on the fly
        idx = kv_cache.pos_idx if kv_cache else 0

        q, k = map(lambda t: self._rope_view(t, idx), (q, k))

        # KV cache for faster inference
        if kv_cache:
            k, v = kv_cache(k, v)

        # Efficient attention implementation
        # ... -> (B, H, S, D / H)
        if FLEX_ATTENTION:
            z = flex_attention(q, k, v, block_mask=mask, enable_gqa=True)
        else:
            k, v = map(lambda t: torch.repeat_interleave(t, dim=2, repeats=self.heads_per_group), (k, v))
            is_causal = mask is not None
            z = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

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
        - seq_len: sequence length
        - dim: embedding dimension
        - theta: rope angle parameter

        ### Returns
        - rope_modulator: tensor of shape (seq_len, dim, 2) whose (t, k) element is
            .. math::
                \\cos(\\frac{2 \\pi t}{\\theta^{(2k / d)}}),
                \\sin(\\frac{2 \\pi t}{\\theta^{(2k / d)}})
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim - 1, 2) / dim))
        t = torch.arange(seq_len) * 2 * math.pi
        angles = torch.outer(t, freqs)
        cos, sin = angles.cos(), angles.sin()
        return torch.stack((cos, sin), dim=-1)

    def _rope_view(self, qk: torch.Tensor, idx: int = 0) -> torch.Tensor:
        """
        Recast tensor to complex numbers and apply rotational position filter.

        ### Parameters
        - qk: query or key tensor
        - idx: position in sequence
        """
        B, H, S, dim = qk.size()
        assert S <= self.rope_modulator.size(0), "sequence length is too long for rope attention"

        rm = self.rope_modulator[idx : idx + S].view(1, 1, S, dim // 2, 2)
        qk = qk.reshape(B, H, S, dim // 2, 2)

        # # (x1 * cos - x2 * sin, x2 * cos + x1 * sin)
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
    - config: configuration class containing arguments for SelfAttention and FeedForward
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

    def weight_initialization(self, init_std: float, factor: float) -> None:
        self.attn.reset_parameters(init_std, factor)
        self.attn_norm.reset_parameters()
        self.ffn.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()

    def forward(self, x: torch.Tensor, kv_cache: KVCache = None, mask: BlockMask = None) -> torch.Tensor:
        out = x + self.attn(self.attn_norm(x), kv_cache=kv_cache, mask=mask)
        out = out + self.ffn(self.ffn_norm(out))
        return out

    def get_nb_flop(self, mode: str = "both", seq_len: int = None) -> int:
        """
        TODO
        Number of flop to process a new token

        ### Parameters
        - mode: whether to consider the forward, backward pass or both
        - seq_len: sequence length
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

        # default to no group query
        if not self.block.nb_kv_heads:
            self.block.nb_kv_heads = self.block.nb_heads


class Transformer(BlockLanguageModel):
    """Decoder only transformer."""

    def __init__(self, config: TransformerConfig):
        super().__init__(config, block=TransformerBlock)

        # cache size
        self.seq_len = config.block.seq_len
        self.head_dim = config.block.emb_dim // config.block.nb_heads
        self.nb_kv_heads = config.block.nb_kv_heads

        # default cache and mask
        self.kv_caches = [None for _ in range(config.nb_layers)]
        self.default_mask = build_attention_mask("causal", seq_len=self.seq_len)

    def build_mask(self, x: torch.Tensor) -> BlockMask:
        """
        Build attention mask for the input tensor.

        ### Parameters
        - x: input tensor

        ### Returns
        - mask: attention mask
        """
        # during pretraining all sequences are of the same length, return default mask
        if x.size(1) == self.seq_len:
            return self.default_mask.to(x.device)

        # at inference time, we pass tokens one by one
        elif x.size(1) == 1:
            return None

        # otherwise, we assume that we are in prefilling mode
        else:
            return build_attention_mask("causal", seq_len=x.size(1)).to(x.device)

    def build_cache(self, bsz: int) -> KVCache:
        # build cache
        cache_size = (bsz, self.nb_kv_heads, self.seq_len, self.head_dim)
        dtype = self.embeddings.weight.dtype
        device = self.embeddings.weight.device
        self.kv_caches = [
            KVCache(
                torch.zeros(cache_size, dtype=dtype, device=device),
                torch.zeros(cache_size, dtype=dtype, device=device),
            )
            for _ in self.layers
        ]

    def forward(self, x: torch.Tensor, mask: BlockMask = None) -> torch.Tensor:
        if mask is None:
            mask = self.build_mask(x)
        out = self.embeddings(x)
        for layer, kv_cache in zip(self.layers, self.kv_caches):
            out = layer(out, kv_cache=kv_cache, mask=mask)
        logits = self.output(self.output_norm(out))
        return logits
