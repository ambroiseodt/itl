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
from types import TracebackType
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import yaml
from torch.distributed.checkpoint.stateful import Stateful

from src.nanollama.data.tokenizer import TokenizerConfig, build_tokenizer
from src.nanollama.distributed import ClusterConfig, ClusterManager, get_rank, is_master_process
from src.nanollama.inference import QueuedBatchedInference
from src.nanollama.model import (
    BlockLanguageModel,
    build_config_with_model_dispatch,
)
from src.nanollama.monitor import (
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


class EvalCheckpointer:
    """
    Automatically save and load evaluation state to avoid repeated computation.

    ### Parameters
    - tmp_file: temporary file to store partial results
    - eval_state: evaluation state
    """

    def __init__(self, tmp_file: str, eval_state: EvalState):
        self.tmp_file = Path(tmp_file)
        self.state = eval_state

    def __enter__(self):
        # retrieve previous computations
        if self.tmp_file.exists():
            with open(os.path.expandvars(self.tmp_file)) as f:
                state = json.load(f)
            self.state.load_state_dict(state)
        else:
            self.tmp_file.touch()
        return self

    def delete(self) -> None:
        self.tmp_file.unlink()

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        if self.tmp_file.exists():
            with open(os.path.expandvars(self.tmp_file), "w") as f:
                print(json.dumps(self.state.state_dict()), file=f, flush=True)


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

    log_path: str
    db_path: str
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    tmp_file: str = ""

    def __post_init__(self):
        self.log_path = os.path.expandvars(self.log_path)
        self.db_path = os.path.expandvars(self.db_path)
        self.tmp_file = os.path.expandvars(self.tmp_file)

        if not self.tmp_file:
            self.tmp_file = Path(self.log_path).parent / f".tmp_eval_{get_rank()}.jsonl"


@torch.inference_mode()
def run_evaluation(config: OnlineEvaluationConfig, model: BlockLanguageModel, preemption: PreemptionHandler = None, **metadata) -> None:
    if preemption is None:
        preemption = lambda: False

    with ExitStack() as context_stack:
        state = EvalState()

        # partial evaluation checkpointer
        checkpointer = EvalCheckpointer(config.tmp_file, state)
        checkpointer: EvalCheckpointer = context_stack.enter_context(checkpointer)

        # data loader
        loader = PromptLoader(config.data, dp_mesh=None)
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
        with open(config.log_path, "a") as f:
            print(json.dumps({"loss": state.loss} | metadata), file=f, flush=True)

        checkpointer.delete()

    return state.loss


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

    checkpoint_dir: str = "/private/home/vivc/memory/checkpoints/0/0000010000"

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    orchestration: EvalOrchestratorConfig = field(default_factory=EvalOrchestratorConfig)

    def __post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        # path to stored results
        self.path = self.orchestration.logging.metric_path

        self.log_path = ...
        self.db_path = ...
        self.tmp_file = ...

        super().__post_init__()

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
        # Build and Parallelize model
        # ---------------------------------------------------------------------

        # build model architecture
        with open(f"{config.checkpoint_dir}/params.json") as f:
            model_config = {"model": json.load(f)}
        model_config = build_config_with_model_dispatch(None, model_config)
        model: BlockLanguageModel = model_config.model_gen(model_config.model)

        # parallelize model
        model = cluster.build_model(model)

        # load model weights
        state_dict = {"model": model.state_dict()}
        dcp.load(state_dict=state_dict, checkpoint_id=config.checkpoint_dir)
        model.load_state_dict(state_dict["model"])

        # ---------------------------------------------------------------------
        # Recover Checkpoint
        # ---------------------------------------------------------------------

        train_step = config.orchestration.train_step
        context_stack.enter_context(EvalCheckpointer(model, config.orchestration.checkpoint_path, train_step))

        # ---------------------------------------------------------------------
        # Run evaluation into chunks
        # ---------------------------------------------------------------------

        metric = run_evaluation(config, model, preemption=preemption, step=train_step)

        # wandb logging
        wandb: WandbLogger = context_stack.enter_context(WandbLogger(config.orchestration.wandb, config))
        wandb({"test_loss": metric, "step": train_step})

        if is_master_process():
            logger.info(f"Test loss: {round(metric, 4):>7}")

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
