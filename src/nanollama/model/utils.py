# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Utilities to load models

@ 2025, Meta
"""
from dataclasses import dataclass, field
from typing import Any

from ..utils import initialize_nested_object
from .transformer import Transformer, TransformerConfig


def build_config_with_model_dispatch(ConfigClass: type, run_config: dict[str, Any]) -> Any:
    """
    Initialize configuration based on the model implementation specified in the run_config.

    ### Parameters
    - ConfigClass: The configuration class to be used.
    - run_config: A dictionary containing the configuration details.

    ### Returns
    - config: The initialized configuration object.
    """
    # argument parsing
    assert "model" not in run_config and "implementation" in run_config["model"], "Model implementation not found"
    implementation = run_config["model"]["implementation"]

    if ConfigClass is None:

        @dataclass
        class ConfigClass:
            pass

    match implementation:
        case "transformer":

            @dataclass
            class Config(ConfigClass):
                model: TransformerConfig = field(default_factory=TransformerConfig)
                model_gen: callable = field(init=False, default=Transformer)

        case "mamba":
            from src.nanollama.model.mamba import Mamba, MambaConfig

            @dataclass
            class Config(ConfigClass):
                model: MambaConfig = field(default_factory=MambaConfig)
                model_gen: callable = field(init=False, default=Mamba)

        case "hawk":
            from src.nanollama.model.rnn import FastRNNConfig, Hawk

            @dataclass
            class Config(ConfigClass):
                model: FastRNNConfig = field(default_factory=FastRNNConfig)
                model_gen: callable = field(init=False, default=Hawk)

        case "mingru":
            from src.nanollama.model.rnn import FastRNNConfig, MinGRU

            @dataclass
            class Config(ConfigClass):
                model: FastRNNConfig = field(default_factory=FastRNNConfig)
                model_gen: callable = field(init=False, default=MinGRU)

        case "minlstm":
            from src.nanollama.model.rnn import FastRNNConfig, MinLSTM

            @dataclass
            class Config(ConfigClass):
                model: FastRNNConfig = field(default_factory=FastRNNConfig)
                model_gen: callable = field(init=False, default=MinLSTM)

        case _:
            raise ValueError(f"Model implementation {implementation} not found")

    # Initialize configuration
    config = initialize_nested_object(Config, run_config)
    return config
