# This source code is licensed under the terms specified in the `LICENSE` file.
"""
RNN utilities

@ 2025, Meta
"""

from dataclasses import dataclass, field
from typing import Literal

from .embedding_model import EmbeddingModel, EmbeddingModelConfig
from .ssm.hawk import HawkBlock
from .ssm.mingru import GRUBlock
from .ssm.minlstm import LSTMBlock
from .ssm.utils_rnn import RNNBlockConfig

# ------------------------------------------------------------------------------
# Configuration class (see ssm.rnn_utils.RNNBlockConfig)
# ------------------------------------------------------------------------------


@dataclass
class FastRNNConfig(EmbeddingModelConfig):
    implementation: Literal["minlstm", "mingru", "hawk"]
    block: RNNBlockConfig = field(default_factory=RNNBlockConfig)

    def post_init(self) -> None:
        super().post_init()

        # inherit parameters from the block model configuration
        for attr in ["emb_dim"]:
            setattr(self.block, attr, getattr(self, attr))

        # default scaling of hidden dimensions
        if self.implementation == "hawk":
            if not self.block.hidden_dim:
                self.block.hidden_dim = 4 * self.emb_dim
            if not self.block.ffn_dim:
                self.block.ffn_dim = 4 * self.emb_dim
        else:
            if not self.block.hidden_dim:
                self.block.hidden_dim = 3 * self.emb_dim

        self.block.post_init()


# ------------------------------------------------------------------------------
# Various Architectures
# ------------------------------------------------------------------------------


class Hawk(EmbeddingModel):
    def __init__(self, config: FastRNNConfig) -> None:
        super().__init__(config, block=HawkBlock)


class MinGRU(EmbeddingModel):
    def __init__(self, config: FastRNNConfig) -> None:
        super().__init__(config, block=GRUBlock)


class MinLSTM(EmbeddingModel):
    def __init__(self, config: FastRNNConfig) -> None:
        super().__init__(config, block=LSTMBlock)
