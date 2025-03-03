# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Utilities to load models

@ 2025, Meta
"""

from .blocklm import BlockLanguageModel, BlockLanguageModelConfig
from .transformer import Transformer, TransformerConfig


def build_config(implementation: str) -> tuple[BlockLanguageModelConfig, BlockLanguageModel]:
    """
    Return the configuration and model class for the given implementation

    ### Parameters
    - implementation: str

    ### Returns
    - config: BlockLanguageModelConfig
    - model: BlockLanguageModel
    """
    match implementation:
        case "transformer":
            config = TransformerConfig
            model = Transformer

        case "mamba":
            from .mamba import Mamba, MambaConfig

            config = MambaConfig
            model = Mamba

        case "hawk":
            from .rnn import FastRNNConfig, Hawk

            config = FastRNNConfig
            model = Hawk

        case "mingru":
            from .rnn import FastRNNConfig, MinGRU

            config = FastRNNConfig
            model = MinGRU

        case "minlstm":
            from .rnn import FastRNNConfig, MinLSTM

            config = FastRNNConfig
            model = MinLSTM

    return config, model
