# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Utilities to load models

@ 2025, Meta
"""
from typing import Any

from ..utils import build_with_type_check
from .transformer import Transformer, TransformerConfig


def build_model_config(model_config: dict[str, Any]) -> tuple[object, type]:
    """
    Initialize configuration based on the model implementation specified in the run_config.

    ### Parameters
    - ConfigClass: The configuration class to be used.
    - run_config: A dictionary containing the configuration details.

    ### Returns
    - config: The initialized configuration object.
    """
    # argument parsing
    implementation = model_config.get("implementation", "transformer")

    match implementation:
        case "transformer":
            model = Transformer
            config = build_with_type_check(TransformerConfig, model_config)

        case "mamba":
            from src.nanollama.model.mamba import Mamba, MambaConfig

            model = Mamba
            config = build_with_type_check(MambaConfig, model_config)

        case "hawk":
            from src.nanollama.model.rnn import FastRNNConfig, Hawk

            model = Hawk
            config = build_with_type_check(FastRNNConfig, model_config)

        case "mingru":
            from src.nanollama.model.rnn import FastRNNConfig, MinGRU

            model = MinGRU
            config = build_with_type_check(FastRNNConfig, model_config)

        case "minlstm":
            from src.nanollama.model.rnn import FastRNNConfig, MinLSTM

            model = MinLSTM
            config = build_with_type_check(FastRNNConfig, model_config)

        case _:
            raise ValueError(f"Model implementation {implementation} not found")

    return config, model
