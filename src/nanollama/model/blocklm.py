# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Model as blocks acting in an embedding space.

@ 2025, Meta
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .norm import RMSNorm


class BlockModel(nn.Module, ABC):
    """
    Abstract class for models
    """

    @abstractmethod
    def get_nb_flop(self, mode: str = "both", **kwargs) -> None:
        """
        Number of flop to process a new token

        ### Parameters
        - mode: whether to consider the forward, backward pass or both
        """
        ...

    @abstractmethod
    def reset_parameters(self, init_std: float, factor: float) -> None:
        """
        Weight initialization of submodules

        ### Parameters
        - init_std: standard deviation of the initialization
        - factor: scaling factor for the output layer
        """
        ...


@dataclass
class BlockLanguageModelConfig:
    # Block configuration
    implementation: str
    block: Any

    # Embedding parameters
    emb_dim: int = 0
    vocab_size: int = 0

    # Model parameters
    nb_layers: int = 0
    weight_tying: bool = False
    norm_eps: float = 1e-5
    init_std: float = None

    def post_init(self) -> None:
        assert self.block, "block should be specified"
        assert self.emb_dim, "embedding dimension should be specified"
        assert self.nb_layers, "number of layers should be specified"
        assert self.vocab_size, "vocabulary size should be specified"


class BlockLanguageModel(ABC, nn.Module):
    """
    Language model based on block acting in an embedding space.
    """

    def __init__(self, config: BlockLanguageModelConfig, block: BlockModel):
        super().__init__()

        self.emb_dim = config.emb_dim
        self.weight_tying = config.weight_tying

        self.embeddings = nn.Embedding(config.vocab_size, config.emb_dim)

        self.layers = nn.ModuleList([block(config.block) for _ in range(config.nb_layers)])

        self.output = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        self.output_norm = RMSNorm(config.emb_dim, eps=config.norm_eps)

        if config.weight_tying:
            # Tying token embedding and un-embedding
            self.output.weight = self.embeddings.weight

        self.reset_parameters(config.init_std, factor=1.0)

    @property
    def device(self) -> torch.device:
        return self.embeddings.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.weight.dtype

    @torch.inference_mode()
    def reset_parameters(self, init_std: float, factor: float) -> None:
        """
        Resetting module parameters

        ### Parameters
        - init_std: standard deviation of the initialization
        - factor: scaling factor for the output layer
        """
        emb_std = init_std or (self.emb_dim ** (-0.5))

        # embeddings
        nn.init.trunc_normal_(
            self.embeddings.weight,
            mean=0.0,
            std=emb_std,
            a=-3 * emb_std,
            b=3 * emb_std,
        )

        # output
        self.output_norm.reset_parameters()
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=emb_std,
                a=-3 * emb_std,
                b=3 * emb_std,
            )

        # layers
        for layer in self.layers:
            layer: BlockModel
            layer.reset_parameters(init_std, factor=factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.embeddings(x)
        for layer in self.layers:
            out = layer(out)
        logits = self.output(self.output_norm(out))
        return logits

    def get_nb_flop(self, mode: str = "both", **kwargs) -> int:
        """
        TODO
        Number of flop to process a new token

        ### Parameters
        - mode: whether to consider the forward, backward pass or both
        - kwargs: argument to compute the number of flops of the block
        """
        mode_multiplier = dict(fwd=1, bwd=2.5, both=3.5)[mode]
        return 0 * mode_multiplier

    @abstractmethod
    def build_cache(self, bsz: int) -> None:
        """
        Build model cache for autoregressive inference

        ### Parameters
        - bsz: batch size
        """
        ...

    @abstractmethod
    def delete_cache(self) -> None:
        """
        Delete model cache
        """
        ...
