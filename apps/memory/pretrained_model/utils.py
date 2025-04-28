# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Utilities for configuration, argument parsing, loading, and inference with HuggingFace pretrained models.

@ 2025, Meta


### Notes
For consistency, we use a logic similar to that of src/nanollama/model/transformer/inference.py
but note that the generate() method from HuggingFace transformers library combines optimized version of
the functions  implemented in this file. Our implementation trades performance for simplicity of use.
"""

from dataclasses import asdict, dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import ModelOutput

from src.nanollama.distributed import ClusterConfig
from src.nanollama.launcher import LauncherConfig
from src.nanollama.monitor import (
    Checkpointer,
    OrchestratorConfig,
)
from src.nanollama.optim import (
    OptimizerConfig,
)
from src.nanollama.tokenizer import DialogTokenizer
from src.nanollama.utils import build_with_type_check

from ..args import (
    EvalConfig,
    MemoryDataConfig,
    OnlineEvaluationConfig,
    heritage_eval_config,
    heritage_grid_id,
    heritage_launch_config,
)

logger = getLogger("nanollama")

# ------------------------------------------------------------------------------
# Evaluation Configuration for Finetuning runs
# ------------------------------------------------------------------------------


def build_eval_finetuning_launch_config(
    eval_config: EvalConfig, orch: OrchestratorConfig, checkpoint: Checkpointer, step: int
) -> tuple[LauncherConfig, dict[str, Any]]:
    eval_flag = "flag"
    checkpoint.update(eval_flag=eval_flag)
    step_id = checkpoint.folder_name.format(step)

    eval_orch = eval_config.orchestration

    # specify training step etc.
    eval_config.log_path = str(Path(orch.logging.metric_path) / "eval_0.jsonl")
    eval_config.model_dir = str(Path(orch.checkpoint.path) / step_id)
    eval_config.checkpoint_flag = eval_flag
    eval_config.metadata = {"step": step}
    eval_orch.log_dir = str(Path(eval_orch.log_dir) / step_id)

    # launcher config
    eval_config.slurm.post_init()
    launch_config = build_with_type_check(
        LauncherConfig,
        {
            "name": eval_orch.name,
            "log_dir": eval_orch.log_dir,
            "overwrite": False,
            "copy_code": False,
            "script": "apps.memory.pretrained_model.eval",
            "slurm": asdict(eval_config.slurm),
        },
    )

    # check and format config
    eval_dict = asdict(eval_config)
    eval_dict.pop("period")
    eval_dict.pop("asynchronous")
    eval_dict.pop("slurm")
    run_config = {"run_config": eval_dict}

    # remove step from log_dir
    eval_orch.log_dir = str(Path(eval_orch.log_dir).parent)

    return launch_config, run_config


# ------------------------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------------------------


@dataclass
class FinetuningConfig:
    compile: bool = False
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    orchestration: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    data: MemoryDataConfig = field(default_factory=MemoryDataConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # to allow for various implementation, we do not type checked model and tokenizer here
    model: dict[str, Any] = field(default_factory=dict)
    tokenizer: dict[str, Any] = field(default_factory=dict)
    model_callback: callable = field(init=False, default=None)

    def __post_init__(self) -> None:
        """
        Check validity of arguments and fill in missing values.
        """
        # restriction for cpu run
        self.cluster.post_init()
        if torch.device(self.cluster.device).type == "cpu":
            assert self.optim.fused is False, "Fused Adam is not supported on CPU"
            assert self.orchestration.profiler.pytorch.active is False, "Profiler is not supported on CPU"

        # manual post initialization of all modules
        self.orchestration.post_init()
        self.data.post_init()
        self.optim.post_init()

        # only post init evaluation if online
        eval = self.evaluation
        if eval.period and not eval.asynchronous:
            OnlineEvaluationConfig.post_init(self.evaluation)


def build_finetune_config(file_config: dict[str, Any]) -> FinetuningConfig:
    """
    Build configuration from file configuration for finetuning run.

    ### Parameters
    - file_config: configuration as a dictionary.

    ### Returns
    - config: configuration as a dataclass.
    """

    if "run_config" in file_config:
        run_config: dict[str, Any] = file_config.pop("run_config")
    else:
        run_config = file_config
    launcher: dict[str, Any] = file_config.pop("launcher", {})

    heritage_launch_config(run_config, launcher)
    heritage_eval_config(run_config, launcher)

    # grid id system to handle special grid cases
    grid_id = run_config.get("grid_id", None)
    if grid_id is not None:
        run_config = heritage_grid_id(run_config, grid_id)

    config = build_with_type_check(FinetuningConfig, run_config)
    return config


# ------------------------------------------------------------------------------
# Build pretrained model from config
# ------------------------------------------------------------------------------


def build_model(config: dict[str, Any], tokenizer: DialogTokenizer, return_config: bool = False) -> nn.Module:
    """
    Load pretrained model based on the specified model implementation.

    ### Parameters
    - config: A dictionary containing the model id to load from HuggingFace.
    - return_config: A boolean indicating whether to return the configuration object.

    ### Returns
    - model: A pretrained HuggingFace model.
    """
    # argument parsing
    model_id = config.get("model_id", "gpt2").lower()
    model_config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Update token embeddings to manage new tokens
    model.resize_token_embeddings(tokenizer.vocab_size)

    # Set to training mode
    model.train()

    if return_config:
        return model, model_config
    return model


# ------------------------------------------------------------------------------
# Sample logits
# ------------------------------------------------------------------------------


def sample_from_logits(logits: Tensor, **kwargs) -> Tensor:
    """Sample from logits."""
    return logits.argmax(dim=-1)


# ------------------------------------------------------------------------------
# Pretraining
# ------------------------------------------------------------------------------


def pretrain(model: nn.Module, x: Tensor, y: Tensor, loss_mask: Tensor = None) -> Tensor:
    """
    Pretraining logic.

    ### Parameters
    - model: transformer model to be pretrained.
    - x: input tensor.
    - y: target tensor.
    - loss_mask: mask for the cross-entropy loss.

    ### Returns
    - loss: cross-entropy loss value.
    """
    # forward propagation
    preds = model(x)

    # compatibility with HuggingFace models
    if isinstance(preds, ModelOutput):
        if hasattr(preds, "logits"):
            preds = preds.logits
        elif hasattr(preds, "last_hidden_state"):
            preds = preds.last_hidden_state
    vocab_size = preds.size(-1)

    # loss on the tokens that need to be generated by the LLM
    if loss_mask is not None:
        preds = preds[loss_mask]
        y = y[loss_mask]

    # reshaping
    preds = preds.reshape(-1, vocab_size)
    y = y.reshape(-1)
    return F.cross_entropy(preds, y)


# ------------------------------------------------------------------------------
# Token Generation
# ------------------------------------------------------------------------------


@torch.inference_mode()
def generate(model: nn.Module, x: Tensor, **kwargs) -> Tensor:
    """
    Generate tokens from a transformer model.

    ### Parameters
    - model: transformer model to generate tokens from.
    - x: input tensor.
    - kwargs: additional arguments for sampling first tokens.

    ### Returns
    - tokens: generated tokens.
    """
    logits = model(x)

    # compatibility with HuggingFace models
    if isinstance(logits, ModelOutput):
        if hasattr(logits, "logits"):
            logits = logits.logits
        elif hasattr(logits, "last_hidden_state"):
            logits = logits.last_hidden_state

    return sample_from_logits(logits, **kwargs)


# ------------------------------------------------------------------------------
# Prefilling
# ------------------------------------------------------------------------------


@torch.inference_mode()
def prefill(model: nn.Module, x: Tensor, **kwargs) -> Tensor:
    """
    Prefill caches based on prompts.

    ### Parameters
    - model: transformer model to prefill.
    - x: input tensor.
    - kwargs: additional arguments for sampling first tokens.
    """
    logits = model(x)

    # compatibility with HuggingFace models
    if isinstance(logits, ModelOutput):
        if hasattr(logits, "logits"):
            logits = logits.logits
        elif hasattr(logits, "last_hidden_state"):
            logits = logits.last_hidden_state

    # only sample the last token
    logit = logits[:, -1:]
    return sample_from_logits(logit, **kwargs)
