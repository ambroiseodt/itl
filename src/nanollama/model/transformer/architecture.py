# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Transformer model.

@ 2025, Meta
"""

# Comments abbreviations:
#     B: batch size (also bsz)
#     S: sequence length (also seq_len)
#     D: embedding dimension (also emb_dim)
#     H: number of heads (also nb_heads)

import math
from dataclasses import dataclass, field
from logging import getLogger
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..embedding_model import EmbeddingModel, EmbeddingModelConfig
from ..feedforward import FeedForward
from ..norm import RMSNorm

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# Attention Cache
# ------------------------------------------------------------------------------


class KVCache(nn.Module):
    """
    Key-Value cache for faster inference.

    ### Parameters
    - shape: shape of the cache in terms of (bsz, nb_kv_heads, seq_len, head_dim)
    - dynamic: whether to use dynamic shape for the attention mechanisms

    ### Attributes
    - pos_idx: current position in the cache
    - key: prefilled key tensor
    - value: prefilled value tensor
    """

    def __init__(self, shape: list[int], device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.pos_idx = 0
        self.register_buffer("key", torch.zeros(shape, device=device, dtype=dtype))
        self.register_buffer("value", torch.zeros(shape, device=device, dtype=dtype))
        self.key: Tensor
        self.value: Tensor

    def forward(self, key: Tensor, value: Tensor) -> None:
        """
        Update KV cache with new key and value tensors.

        ### Parameters
        - key: key tensor
        - value: value tensor
        - pos_idx: position index to update the cache

        ### Returns
        - key: updated key tensor (concatenating past values with new ones)
        - value: updated value tensor
        """
        # Sequence length is the third dimension, (B, H, S, D / H)
        bsz, _, seq_len, _ = key.size()
        self.key[:bsz, :, self.pos_idx : self.pos_idx + seq_len] = key
        self.value[:bsz, :, self.pos_idx : self.pos_idx + seq_len] = value
        self.pos_idx = self.pos_idx + seq_len
        return self.key[:bsz, :, : self.pos_idx], self.value[:bsz, :, : self.pos_idx]

    def reset(self) -> None:
        self.pos_idx = 0
        # zero out values to gain from sparse kernels
        self.key.zero_()
        self.value.zero_()


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
        qkv_bias: bool,
    ):
        super().__init__()

        # dimensions
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.head_dim = self.emb_dim // nb_heads
        assert self.emb_dim % nb_heads == 0, "embedding dimension must be divisible by number of heads"
        assert nb_heads % nb_kv_heads == 0, "number of heads must be divisible by number of key-value heads"

        # matrices
        self.W_query = nn.Linear(self.emb_dim, self.head_dim * nb_heads, bias=qkv_bias)
        self.W_key = nn.Linear(self.emb_dim, self.head_dim * nb_kv_heads, bias=qkv_bias)
        self.W_val = nn.Linear(self.emb_dim, self.head_dim * nb_kv_heads, bias=qkv_bias)
        self.W_out = nn.Linear(nb_heads * self.head_dim, self.emb_dim, bias=False)

        # rotational positional encoding
        self.theta = rope_theta
        self.register_buffer(
            "rope_modulator", self._get_rope_modulator(self.seq_len, self.head_dim, self.theta), persistent=False
        )
        self.rope_modulator: Tensor

        # inference utilities
        self.kv_cache: KVCache = None
        self.mode: Literal["train", "prefill", "generate"] = "train"

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
            if W_in.bias is not None:
                nn.init.constant_(W_in.bias, 0.0)

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

    def forward(self, x: Tensor) -> Tensor:
        # dimensions
        bsz, seq_len, _ = x.size()

        # Query, key, value: (B, S, D) @ (D, D) -> (B, S, D)
        q, k, v = self.W_query(x), self.W_key(x), self.W_val(x)

        # reformating: (B, S, D) -> (B, S, H, D / H) -> (B, H, S, D / H)
        q, k, v = map(lambda t: t.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2), (q, k, v))

        # rope formatting
        # ... retrieve position in sequence if generating on the fly
        pos_idx = self.kv_cache.pos_idx if self.kv_cache is not None else 0

        q, k = map(lambda t: self._rope_view(t, pos_idx), (q, k))

        # KV cache for faster inference
        if self.kv_cache is not None:
            k, v = self.kv_cache(k, v)

        # Efficient attention implementation (one may use flex attention for fancy masks)
        # ... -> (B, H, S, D / H)
        if self.mode == "train":
            z = self._train_attention(q, k, v)
        elif self.mode == "prefill":
            z = self._prefill_attention(q, k, v)
        elif self.mode == "generate":
            z = self._generate_attention(q, k, v)
        else:
            raise ValueError(f"unknown mode: {self.mode}")

        # reformating: (B, H, S, D / H) -> (B, S, H, D / H) -> (B, S, D)
        z = z.transpose(1, 2).reshape(bsz, seq_len, -1)

        # output layer: (B, L, D) @ (D, D) -> (N, L, D)
        z = self.W_out(z)
        return z

    def _train_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    def _prefill_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # current implement without sequence concatenation nor static caching
        # not that an implementation change may require a change of the cache logic, and of the forward inputs
        assert q.size(2) == k.size(2), "you are breaking our prefilling logic"
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    def _generate_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # current implement without sequence concatenation nor static caching
        assert q.size(2) == 1, "you are breaking our generation logic"
        return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    @staticmethod
    def _get_rope_modulator(seq_len: int, dim: int, theta: float) -> Tensor:
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

    def _rope_view(self, qk: Tensor, idx: int = 0) -> Tensor:
        """
        Recast tensor to complex numbers and apply rotational position filter.

        ### Parameters
        - qk: query or key tensor
        - idx: position in sequence
        """
        bsz, nb_heads, seq_len, dim = qk.size()
        assert seq_len <= self.rope_modulator.size(0), "sequence length is too long for rope attention"

        cos_sin = self.rope_modulator[idx : idx + seq_len].view(1, 1, seq_len, dim // 2, 2)
        qk = qk.reshape(bsz, nb_heads, seq_len, dim // 2, 2)

        # # (x1 * cos - x2 * sin, x2 * cos + x1 * sin)
        out = torch.stack(
            [
                qk[..., 0] * cos_sin[..., 0] - qk[..., 1] * cos_sin[..., 1],
                qk[..., 0] * cos_sin[..., 1] + qk[..., 1] * cos_sin[..., 0],
            ],
            dim=-1,
        )

        return out.type_as(qk).view((bsz, nb_heads, seq_len, dim))

    def get_nb_flop(self, mode: str = "both") -> int:
        """
        Number of flop to process a new token

        ### Notes
        The following formula is somewhat sketchy, it would be nice to validate it empirically.
        This could be done by using the torch dispatcher
        https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505

        ### Parameters
        - mode: whether to consider the forward, backward pass or both
        """
        # flops due to matrix multiplication
        mat_flops = 2 * 2 * self.W_query.weight.numel()
        mat_flops += 2 * 2 * self.W_key.weight.numel()
        mode_multiplier = dict(fwd=1, bwd=2, both=3)[mode]
        flops = mode_multiplier * mat_flops

        # flops due to attention
        # adapated from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
        attn_flops = 2 * self.seq_len * self.W_query.weight.shape[1]
        mode_multiplier = dict(fwd=1, bwd=2.5, both=3.5)[mode]
        flops += mode_multiplier * attn_flops
        return flops


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
    qkv_bias: bool = False

    def post_init(self) -> None:
        """Check validity of arguments that may have been inherited."""
        assert self.seq_len, "sequence length should be specified"
        assert self.emb_dim, "embedding dimension should be specified"
        assert self.hidden_dim, "hidden dimension should be specified"
        assert self.nb_heads, "number of heads should be specified"
        assert self.emb_dim // (2 * self.nb_heads), "embedding dimension should be divisible by 2 * number of heads"


# ------------------------------------------------------------------------------
# Transformer Block
# ------------------------------------------------------------------------------


class TransformerBlock(nn.Module):
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
            qkv_bias=config.qkv_bias,
        )
        self.ffn = FeedForward(emb_dim=config.emb_dim, hidden_dim=config.hidden_dim)
        self.attn_norm = RMSNorm(config.emb_dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.emb_dim, eps=config.norm_eps)

    def reset_parameters(self, init_std: float, factor: float) -> None:
        self.attn.reset_parameters(init_std, factor)
        self.attn_norm.reset_parameters()
        self.ffn.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        out = x + self.attn(self.attn_norm(x))
        out = out + self.ffn(self.ffn_norm(out))
        return out

    def get_nb_flop(self, mode: str = "both") -> int:
        """
        Number of flop to process a new token

        ### Notes
        The following formula is somewhat sketchy, it notably forgets the normalization layers.

        ### Parameters
        - mode: whether to consider the forward, backward pass or both
        """
        flops = self.attn.get_nb_flop(mode)
        flops += self.attn_norm.get_nb_flop(mode)
        flops += self.ffn.get_nb_flop(mode)
        flops += self.ffn_norm.get_nb_flop(mode)
        return flops


# ------------------------------------------------------------------------------
# Transformer Architecture
# ------------------------------------------------------------------------------


@dataclass
class TransformerConfig(EmbeddingModelConfig):
    implementation: Literal["transformer"] = "transformer"
    block: TransformerBlockConfig = field(default_factory=TransformerBlockConfig)

    def post_init(self) -> None:
        super().post_init()

        # Inherit parameters from the block model configuration.
        for attr in ["emb_dim", "norm_eps"]:
            setattr(self.block, attr, getattr(self, attr))

        # default scaling of ffn dimension
        if not self.block.hidden_dim:
            self.block.hidden_dim = 4 * self.emb_dim

        # default to no group query
        if not self.block.nb_kv_heads:
            self.block.nb_kv_heads = self.block.nb_heads

        # check validity of submodules
        self.block.post_init()


class Transformer(EmbeddingModel):
    """Decoder only transformer."""

    def __init__(self, config: TransformerConfig):
        super().__init__(config, block=TransformerBlock)

        # cache size
        self.seq_len = config.block.seq_len
        self.head_dim = config.block.emb_dim // config.block.nb_heads
        self.nb_kv_heads = config.block.nb_kv_heads

    def set_mode(self, mode: str) -> None:
        assert mode in ["train", "prefill", "generate"], f"unknown mode: {mode}"
        for layer in self.layers:
            layer: TransformerBlock
            layer.attn.mode = mode
