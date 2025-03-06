# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Evaluation script.

@ 2025, Meta
"""

import json
import logging
import os
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
import yaml
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh

from src.nanollama.data.tokenizer import TokenizerConfig, build_tokenizer
from src.nanollama.distributed import ClusterConfig, ClusterManager, is_master_process
from src.nanollama.inference import QueuedBatchedInference
from src.nanollama.model import (
    BlockLanguageModel,
    build_config_with_model_dispatch,
)
from src.nanollama.monitor import (
    EvalCheckpointConfig,
    EvalCheckpointer,
    EvalOrchestratorConfig,
    Logger,
    PreemptionHandler,
    WandbLogger,
)
from src.nanollama.utils import initialize_nested_object

from .prompt_loader import DataConfig, PromptLoader

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# Evaluation State
# ------------------------------------------------------------------------------


@dataclass
class EvalState(Stateful):
    accuracy: float = 0
    scaling: float = 0
    step: int = 0

    def state_dict(self) -> dict:
        return {"accuracy": self.accuracy, "scaling": self.scaling, "step": self.step}

    def load_state_dict(self, state_dict: dict) -> None:
        self.accuracy = state_dict["accuracy"]
        self.scaling = state_dict["scaling"]
        self.step = state_dict["step"]


# ------------------------------------------------------------------------------
# Online Evaluation
# ------------------------------------------------------------------------------

def perplexity_func(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = preds.size(-1)
    return torch.exp(F.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1)))

@dataclass
class OnlineEvaluationConfig:
    """
    Configuration to launch an evaluation during a training run.

    ### Parameters
    - log_path: path to store the evaluation logs.
    - db_path: path to SQL database.
    - data: data configuration.
    - tmp_file: temporary file to store partial results
    """

    db_path: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    checkpoint: EvalCheckpointConfig = field(default_factory=EvalCheckpointConfig)

    def __post_init__(self):
        assert self.db_path, "db_path should be specified."
        self.db_path = os.path.expandvars(self.db_path)

        for module in self.__dict__.values():
            if hasattr(module, "__check_init__"):
                module.__check_init__()


@torch.inference_mode()
def run_evaluation(
    config: OnlineEvaluationConfig,
    model: BlockLanguageModel,
    preemption: PreemptionHandler = None,
    dp_mesh: DeviceMesh = None,
) -> dict[str, Any]:
    """
    Run evaluation and return a dictionary of metrics.

    ### Parameters
    - config: evaluation configuration.
    - model: model to evaluate.
    - preemption: preemption handler.
    - dp_mesh: device mesh for distributed training.

    ### Returns
    - dictionary of metrics.
    """
    if preemption is None:

        def preemption():
            return False

    with ExitStack() as context_stack:
        state = EvalState()

        # partial evaluation checkpointer
        checkpointer = EvalCheckpointer(config.checkpoint, state)
        checkpointer: EvalCheckpointer = context_stack.enter_context(checkpointer)

        # data loader
        loader = PromptLoader(config.data, dp_mesh=dp_mesh)
        loader: PromptLoader = context_stack.enter_context(loader)

        # inference engine
        tokenizer = build_tokenizer(config.tokenizer)
        inference_engine = QueuedBatchedInference(model, tokenizer, config.db_path)
        inference_engine: QueuedBatchedInference = context_stack.enter_context(inference_engine)

        for prompts, answers in loader:
            # handle preemption
            if preemption():
                logger.warning("Preemption flag set")
                break

            outputs = inference_engine.generate(prompts)

            # TODO add evaluation if needed
            bsz = len(prompts)
            accuracy = 0
            for output, answer in zip(outputs, answers):
                accuracy += int(output.endswith(f"{answer}."))
            accuracy /= bsz

            # TODO: double check this scaling (the goal is to end up with the mean of the individual accuracy)
            scaling = bsz / loader.batch_size

            state.accuracy += scaling * accuracy
            state.scaling += scaling
            state.step += 1

            logger.info(f"Evaluation: partial step: {state.step} - accuracy: {round(state.accuracy / state.scaling, 4):>7}")

        # rescale accuracy and save it
        state.accuracy /= state.scaling
        state.scaling = 1

    return {"accuracy": state.accuracy}


# ------------------------------------------------------------------------------
# Online Run
# ------------------------------------------------------------------------------


@dataclass
class EvaluationConfig(OnlineEvaluationConfig):
    """
    Configuration to launch an evaluation during a training run.

    ### Parameters
    - model_dir: path to the model directory.
    - log_path: path to store the evaluation logs.
    - checkpoint_flag: file acting as a flag to avoid deleting checkpoints before evaluation
    - metadata: metadata to add to the evaluation metrics.
    """

    model_dir: str = ""
    log_path: str = ""

    checkpoint_flag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    orchestration: EvalOrchestratorConfig = field(default_factory=EvalOrchestratorConfig)

    def __post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        super().__post_init__()

        assert self.log_path, "log_path must be set."
        assert self.model_dir, "Checkpoint directory must be set."
        self.log_path = os.path.expandvars(self.log_path)

        # manual post initialization of all modules
        for module in self.__dict__.values():
            if hasattr(module, "__check_init__"):
                module.__check_init__()

        # configuration not managed by orchestrator due to inheritance issues
        self.checkpoint.path = str(Path(self.orchestration.log_dir) / "checkpoint")
        if self.checkpoint_flag:
            self.checkpoint.flag = str(Path(self.model_dir) / self.checkpoint_flag)
        self.orchestration.logging.metric_path = self.log_path

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["tokenizer"] = self.tokenizer.to_dict()
        output["cluster"] = self.cluster.to_dict()
        output["orchestration"] = self.orchestration.to_dict()
        return output


@torch.inference_mode()
def eval(config: EvaluationConfig) -> None:
    with ExitStack() as context_stack:
        # ---------------------------------------------------------------------
        # Handle preemption, computing environment, logging, and utils
        # ---------------------------------------------------------------------

        preemption = PreemptionHandler()
        context_stack.enter_context(preemption)

        cluster = ClusterManager(config.cluster)
        context_stack.enter_context(cluster)

        metric_logger = Logger(config.orchestration.logging, eval=True)
        context_stack.enter_context(metric_logger)

        wandb = WandbLogger(config.orchestration.wandb, asdict(config))
        context_stack.enter_context(wandb)

        # ---------------------------------------------------------------------
        # Build, Recover Checkpoint and Parallelize model
        # ---------------------------------------------------------------------

        # build model architecture
        with open(f"{config.model_dir}/params.json") as f:
            model_config = {"model": json.load(f)}
        model_config = build_config_with_model_dispatch(None, model_config)
        model: BlockLanguageModel = model_config.model_gen(model_config.model)

        # load model weights
        state_dict = {"model": model.state_dict()}
        dcp.load(state_dict=state_dict, checkpoint_id=config.model_dir)
        model.load_state_dict(state_dict["model"])

        # parallelize model
        model = cluster.build_model(model)

        # ---------------------------------------------------------------------
        # Evaluation loop
        # ---------------------------------------------------------------------

        metrics = run_evaluation(config, model, preemption=preemption)

        # logging metrics
        metrics |= config.metadata
        metric_logger(metrics)
        wandb(metrics)

        if is_master_process():
            logger.info(f"Evaluation metrics: {metrics}")

    logger.info("Evaluation done.")


# ------------------------------------------------------------------------------
# Main file
# ------------------------------------------------------------------------------


def main() -> None:
    """
    Launch an evaluation job from configuration file specified by cli argument.

    Usage:
    ```
    python -m apps.my_app.evaluation apps/my_app/configs/my_config.yaml
    ```
    """
    import argparse

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # parse file configuration path
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("config", type=str, help="Path to configuration file")
    path = parser.parse_args().config

    # obtain configuration from file
    with open(os.path.expandvars(path)) as f:
        text_config: dict[str, Any] = yaml.safe_load(f)

    if "run_config" in text_config:
        text_config = text_config["run_config"]

    # initialize configuration
    config = initialize_nested_object(EvaluationConfig, text_config, inplace=False)

    # launch job
    eval(config)


if __name__ == "__main__":
    main()
