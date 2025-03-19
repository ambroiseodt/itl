# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Utilities to load models

@ 2025, Meta
"""
from dataclasses import asdict
from typing import Any

from ..utils import build_with_type_check
from .embedding_model import EmbeddingModel
from .transformer import Transformer, TransformerConfig

# ------------------------------------------------------------------------------
# Configuration Dispatcher
# ------------------------------------------------------------------------------


def build_model(config: dict[str, Any], callback: callable = None, return_config: bool = False) -> EmbeddingModel:
    """
    Initialize configuration based on the specified model implementation.

    ### Parameters
    - config: A dictionary containing the configuration details.
    - callback: A callable function to be executed after the configuration is initialized.
    - return_config: A boolean indicating whether to return the configuration object.

    ### Returns
    - model: An instance of the specified model implementation.
    """
    # argument parsing
    implementation = config.get("implementation", "transformer").lower()

    match implementation:
        case "transformer":
            model_type = Transformer
            config_obj = build_with_type_check(TransformerConfig, config)

        case "mamba":
            from src.nanollama.model.mamba import Mamba, MambaConfig

            model_type = Mamba
            config_obj = build_with_type_check(MambaConfig, config)

        case "hawk":
            from src.nanollama.model.rnn import FastRNNConfig, Hawk

            model_type = Hawk
            config_obj = build_with_type_check(FastRNNConfig, config)

        case "mingru":
            from src.nanollama.model.rnn import FastRNNConfig, MinGRU

            model_type = MinGRU
            config_obj = build_with_type_check(FastRNNConfig, config)

        case "minlstm":
            from src.nanollama.model.rnn import FastRNNConfig, MinLSTM

            model_type = MinLSTM
            config_obj = build_with_type_check(FastRNNConfig, config)

        case _:
            raise ValueError(f"Model implementation {implementation} not found")

    # call the callback and post init methods
    if callback is not None:
        callback(config_obj)
    config_obj.post_init()

    # instanciate the model
    model = model_type(config_obj)

    if return_config:
        return model, asdict(config_obj)
    return model
