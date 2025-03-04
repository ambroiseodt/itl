# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Evaluation script.

@ 2025, Meta
"""

import json
import logging
import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
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
    loss: float = 0
    scaling: float = 0
    step: int = 0

    def state_dict(self) -> dict:
        return {"loss": self.loss, "scaling": self.scaling, "step": self.step}

    def load_state_dict(self, state_dict: dict) -> None:
        self.loss = state_dict["loss"]
        self.scaling = state_dict["scaling"]
        self.step = state_dict["step"]


# ------------------------------------------------------------------------------
# Online Evaluation
# ------------------------------------------------------------------------------


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

    db_path: str
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    checkpoint: EvalCheckpointConfig = field(default_factory=EvalCheckpointConfig)

    def __post_init__(self):
        self.db_path = os.path.expandvars(self.db_path)


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
            loss = 0
            for output, answer in zip(outputs, answers):
                loss += int(output.endswith(f"{answer}."))
                print(output)
            loss /= bsz

            # TODO: double check this scaling (the goal is to end up with the mean of the individual loss)
            scaling = bsz / loader.batch_size

            state.loss += scaling * loss
            state.scaling += scaling
            state.step += 1

            logger.debug(f"Evaluation: partial step: {state.step} - loss: {round(state.loss / state.scaling, 4):>7}")

        # rescale loss and save it
        state.loss /= state.scaling
        state.scaling = 1

    return {"loss": state.loss}


# ------------------------------------------------------------------------------
# Online Run
# ------------------------------------------------------------------------------


@dataclass
class EvaluationConfig(OnlineEvaluationConfig):
    """
    Configuration to launch an evaluation during a training run.

    ### Parameters
    - log_path: path to store the evaluation logs.
    - db_path: path to SQL database.
    - data: data configuration.
    - tmp_file: temporary file to store partial results
    """

    model_dir: str = ""
    checkpoint_flag: str = ""
    log_path: str = ""

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

        # checkpoint configuration (not managed by orchestrator due to inheritance issue)
        self.checkpoint.path = str(Path(self.orchestration.log_dir) / "checkpoint")
        if self.checkpoint_flag:
            self.checkpoint.flag = str(Path(self.model_dir) / self.checkpoint_flag)

        # manual post initialization of all modules
        for module in self.__dict__.values():
            if hasattr(module, "__check_init__"):
                module.__check_init__()


@torch.inference_mode()
def eval(config: EvaluationConfig) -> None:
    with ExitStack() as context_stack:
        # ---------------------------------------------------------------------
        # Handle preemption, computing environment, logging, and utils
        # ---------------------------------------------------------------------

        preemption: PreemptionHandler = context_stack.enter_context(PreemptionHandler())
        cluster: ClusterManager = context_stack.enter_context(ClusterManager(config.cluster))
        metric_logger: Logger = context_stack.enter_context(Logger(config.orchestration.logging))

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

        metric = run_evaluation(config, model, preemption=preemption)

        # logging
        step = 0
        metadata = {"step": step}
        metric |= metadata
        with open(config.log_path, "a") as f:
            print(json.dumps(metric), file=f, flush=True)

        wandb: WandbLogger = context_stack.enter_context(WandbLogger(config.orchestration.wandb, config))
        wandb(metric)

        if is_master_process():
            logger.info(f"Test loss: {round(metric['loss'], 4):>7}")

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

    # initialize configuration
    config = initialize_nested_object(EvaluationConfig, text_config, inplace=False)

    # launch job
    eval(config)


if __name__ == "__main__":
    main()
