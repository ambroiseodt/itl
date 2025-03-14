# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Evaluation script.

@ 2025, Meta
"""

import json
import logging
import os
from contextlib import ExitStack
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh

from src.nanollama.data.tokenizer import build_tokenizer
from src.nanollama.distributed import ClusterManager, is_master_process
from src.nanollama.inference import QueuedBatchedInference
from src.nanollama.model import (
    EmbeddingModel,
    build_config_with_model_dispatch,
)
from src.nanollama.monitor import (
    EvalCheckpointer,
    Logger,
    PreemptionHandler,
)

from .args import EvaluationConfig, OnlineEvaluationConfig, build_eval_config
from .prompt_loader import PromptLoader

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

def perplexity_func(preds: Tensor, targets: Tensor) -> Tensor:
    vocab_size = preds.size(-1)
    return torch.exp(F.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1)))


@torch.inference_mode()
def run_evaluation(
    config: OnlineEvaluationConfig,
    model: EmbeddingModel,
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

            # TODO modify inference_engine.generate to return
            # perplexity and/or logits/softmax proba
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

            logger.info(
                f"Evaluation: partial step: {state.step} - accuracy: {round(state.accuracy / state.scaling, 4)}"
            )

        # rescale accuracy and save it
        state.accuracy /= state.scaling
        state.scaling = 1

    return {"accuracy": state.accuracy}


# ------------------------------------------------------------------------------
# Online Run
# ------------------------------------------------------------------------------


@torch.inference_mode()
def eval(config: EvaluationConfig) -> None:
    with ExitStack() as context_stack:
        # ---------------------------------------------------------------------
        # Handle preemption, computing environment and logging
        # ---------------------------------------------------------------------

        preemption = PreemptionHandler()
        context_stack.enter_context(preemption)

        cluster = ClusterManager(config.cluster)
        context_stack.enter_context(cluster)

        metric_logger = Logger(config.orchestration.logging, eval=True)
        context_stack.enter_context(metric_logger)

        # ---------------------------------------------------------------------
        # Build, Recover Checkpoint and Parallelize model
        # ---------------------------------------------------------------------

        # build model architecture
        with open(f"{config.model_dir}/params.json") as f:
            model_config = {"model": json.load(f)}
        model_config = build_config_with_model_dispatch(None, model_config)
        model: EmbeddingModel = model_config.model_gen(model_config.model)

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
        file_config: dict[str, Any] = yaml.safe_load(f)

    # initialize configuration
    config = build_eval_config(file_config)

    # launch job
    eval(config)


if __name__ == "__main__":
    main()
