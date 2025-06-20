# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Evaluation script of HuggingFace pretrained model.

@ 2025, Meta

### Notes
For consistency, we use a logic similar to that of logic similar to that of apps/memory/eval.py
but note that pretrained HuggingFace models have optimized functions for inference and token generation.
Our implementation trades performance for simplicity of use.
"""

import json
import logging
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from logging import getLogger
from types import TracebackType
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import yaml
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh
from transformers import PretrainedConfig

from src.nanollama.agent import Actor, SQLAgent
from src.nanollama.distributed import ClusterManager, get_raw_model, is_master_process
from src.nanollama.monitor import (
    EvalCheckpointer,
    Logger,
    PreemptionHandler,
)
from src.nanollama.tokenizer import DialogTokenizer, build_tokenizer

from ..args import EvaluationConfig, OnlineEvaluationConfig, build_eval_config
from ..prompt_loader import PromptLoader
from .utils import build_model, generate, prefill

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# Generation Logic
# ------------------------------------------------------------------------------


class BatchedInference:
    """
    Batched inference with queues of prefilled tokens to overwrite token generation.

    This implementation is highly inefficient but very simple to work with and debug.

    ### Parameters
    - model: HuggingFace pretrained model
    - model_config: corresponding configuration
    - tokenizer: tokenizer
    - db_path: path to the database containing facts
    - compile: whether to compile pretrain and generation logic
    - kwargs: additional arguments to specify the sampling strategy

    ### Attributes
    - queue: queues of tokens to be added to the context
    """

    def __init__(
        self,
        model: nn.Module,
        model_config: PretrainedConfig,
        tokenizer: DialogTokenizer,
        db_path: str,
        compile: bool = False,
        **kwargs,
    ):
        self.raw_model: nn.Module = get_raw_model(model)
        self.model = model
        self.model_config = model_config

        # Attributes for decoding with agent interaction
        self.tokenizer = tokenizer
        self.queue: list[list[int]]
        self.agent = SQLAgent(db_path)

        # compiled function
        if compile:
            logger.info("Compiling the generation pipeline")
            self.prefilling_logic = torch.compile(prefill, dynamic=True)
            self.generation_logic = torch.compile(generate, dynamic=True)
        else:
            self.prefilling_logic = prefill
            self.generation_logic = generate
        self.kwargs = kwargs

    @property
    def device(self) -> torch.device:
        return self.raw_model.device

    def __enter__(self):
        logger.info("Entering inference context. Spinning up SQL agent")
        self.model.eval()
        self.agent.__enter__()
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        logger.info("Exiting inference context. Shutting down SQL agent")
        self.agent.__exit__(exc, value, tb)
        self.model.train()

    def generate(self, prompts: list[list[int]], max_len: int = None) -> list[str]:
        """
        Generate completions for the given prompts.

        ### Parameters
        - prompts: list of prompts
        - max_len: maximum length of the dialog (completion and prompt)

        ### Returns
        - output: list of completions
        """
        # parse arguments
        if max_len is None:
            max_len = getattr(self.model_config, "n_positions", torch.inf)

        # aliases
        bot2actor = self.tokenizer.bot2actor
        decode = self.tokenizer.tokenizer.decode
        encode = self.tokenizer.tokenizer.encode
        assistant = self.tokenizer.bots[Actor.assistant]
        eod = self.tokenizer.eod
        eod = eod if eod > 0 else None

        # Pack tokens
        x = self.build_batch(prompts)
        prefilled = False

        # keep track of dialog and current message
        output = [x]
        buffers = [[] for _ in prompts]

        # keep track of ongoing lanes
        bsz, total_len = x.size()
        ongoing = torch.ones(bsz, dtype=torch.bool, device=self.device)

        while total_len < max_len and ongoing.any():
            if not prefilled:
                x = self.prefilling_logic(self.model, x, **self.kwargs)
                prefilled = True
            else:
                x = self.generation_logic(self.model, x, **self.kwargs)

            # inspect each lane
            for i in range(bsz):
                # replace token that were set in advanced
                if len(self.queue[i]) > 0:
                    x[i, 0] = self.queue[i].pop(0)
                    continue

                # check if the LLM is calling a tool
                token = x[i, 0].item()

                # check for end of dialog token
                if token == eod:
                    ongoing[i] = False

                actor = bot2actor.get(token, None)

                # if so, decode the current LLM message, seen as instructions
                if actor is not None:
                    instructions = decode(buffers[i])
                    buffers[i] = []
                else:
                    buffers[i].append(token)
                    continue

                # ask the agent to execute itself based on the instructions
                if actor == self.agent.actor:
                    answer = self.agent.execute(instructions)
                    # encode its answer and add it to the queue
                    self.queue[i].extend(encode(answer))
                    # and call assistant turn
                    self.queue[i].append(assistant)

            total_len += 1
            output.append(x)

        output = torch.hstack(output)
        return [self.tokenizer.decode(out.tolist()) for out in output]

    def build_batch(self, prompts: list[str]) -> Tensor:
        """
        Build the batch for the model.

        ### Parameters
        - prompts: list of prompts

        ### Returns
        - input_ids: input tensor
        - batch_offset: position of the first token of each prompt
        """
        bsz = len(prompts)
        seq_len = min([len(datum) for datum in prompts])
        dtype, device = torch.long, self.device

        x = torch.zeros((bsz, seq_len), dtype=dtype, device=device)
        self.queue = [[] for _ in prompts]
        for i, datum in enumerate(prompts):
            x[i, :seq_len] = torch.tensor(datum[:seq_len], dtype=dtype, device=device)
            self.queue[i] = datum[seq_len:]
        return x


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


@torch.inference_mode()
def run_evaluation(
    config: OnlineEvaluationConfig,
    model: nn.Module,
    model_config: PretrainedConfig,
    tokenizer: DialogTokenizer,
    preemption: PreemptionHandler = None,
    dp_mesh: DeviceMesh = None,
) -> dict[str, Any]:
    """
    Run evaluation and return a dictionary of metrics.

    ### Parameters
    - config: evaluation configuration.
    - model: model to evaluate.
    - model_config: corresponding model config
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
        context_stack.enter_context(checkpointer)

        # data loader
        loader = PromptLoader(config.data, tokenizer, dp_mesh=dp_mesh)
        context_stack.enter_context(loader)

        # inference engine
        tokenizer = loader.tokenizer
        inference_engine = BatchedInference(
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            db_path=config.db_path,
        )
        context_stack.enter_context(inference_engine)

        for prompts, answers in loader:
            start = time.time()
            # handle preemption
            if preemption():
                logger.warning("Preemption flag set")
                break

            # generate LLM completion
            outputs = inference_engine.generate(prompts)

            # check accuracy
            bsz = len(prompts)
            accuracy = 0
            for output, answer in zip(outputs, answers):
                accuracy += int(output.endswith(f"{answer}."))
            accuracy /= bsz

            scaling = bsz / loader.batch_size
            state.accuracy += scaling * accuracy
            state.scaling += scaling
            state.step += 1

            throughput = len(outputs) / (time.time() - start)

            logger.info(
                f"Evaluation: partial step: {state.step}, "
                f"accuracy: {round(state.accuracy / state.scaling, 4)}, "
                f"prompt_freq: {round(throughput, 4)}"
            )

        logger.info(f"Generation example:\n {output}")

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
            file_config = json.load(f)
        model, model_config = build_model(file_config, return_config=True)

        # load model weights
        state_dict = {"model": model.state_dict()}
        dcp.load(state_dict=state_dict, checkpoint_id=config.model_dir)
        model.load_state_dict(state_dict["model"])

        # parallelize model
        model = cluster.build_model(model)

        # ---------------------------------------------------------------------
        # Evaluation loop
        # ---------------------------------------------------------------------

        tokenizer = build_tokenizer(config.tokenizer)
        metrics = run_evaluation(config, model, model_config, tokenizer, preemption=preemption)

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
    Launch an evaluation job of a pretrained model from configuration file specified by cli argument.

    Usage:
    ```
    python -m apps.memory.pretrained_model.eval apps/my_app/configs/my_config.yaml
    ```
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
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
