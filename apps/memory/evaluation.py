# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Evaluation script.

@ 2025, Meta
"""

import json
import logging
import os
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from types import TracebackType
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import yaml

from nanollama.data.tokenizer import TokenizerConfig, build_tokenizer
from nanollama.distributed import ClusterConfig, ClusterManager, get_rank, is_master_process  # noqa: F401
from nanollama.inference import QueuedBatchedInference
from nanollama.model import BlockLanguageModel, build_config_with_model_dispatch
from nanollama.utils import initialize_nested_object

from .prompt_loader import DataConfig, PromptLoader

logger = getLogger("nanollama")


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


class Evaluator:
    """
    Evaluation manager

    Running evaluation into chunks in order to handle job preemption.

    Usage:
    ```python
    with EvalComputer(*args) as computer:
        while next(computer):
            pass

    # TODO
    Make it a function, with a context manager for partial saving.
    evaluation_state, eval_checkpointer.
    """

    def __init__(self, model: BlockLanguageModel, config: OnlineEvaluationConfig, **metadata):
        # metadata (e.g. training step)
        self.metadata = metadata

        # file paths
        self.log_path = Path(config.log_path)
        self.tmp_file = Path(config.tmp_file)

        # data loader
        tokenizer = build_tokenizer(config.tokenizer)
        self.loader = PromptLoader(config.data, dp_mesh=None)

        # inference engine
        self.inference_engine = QueuedBatchedInference(model, tokenizer, config.db_path)

    def __enter__(self) -> "Evaluator":
        """Enter evaluation runtime context"""

        # initialize sub-context
        self.inference_engine.__enter__()
        self.loader.__enter__()

        # retrieve previous computations
        if self.tmp_file.exists():
            with open(os.path.expandvars(self.tmp_file)) as f:
                state = json.load(f)
                self.loss = state["loss"]
                self.scaling = state["scaling"]
                self.step = state["step"]
        else:
            self.tmp_file.touch()
            self.loss = 0
            self.scaling = 0
            self.step = 0

        # skip batches that were already evaluated
        self.loader.__enter__()
        for _ in range(self.step):
            next(self.loader)

        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """Enter evaluation runtime context"""

        # if the evaluation was interrupted, save the current state
        if self.tmp_file.exists():
            with open(os.path.expandvars(self.tmp_file), "w") as f:
                print(json.dumps({"loss": self.loss, "scaling": self.scaling, "step": self.step}), file=f, flush=True)

        # exit sub-context
        self.inference_engine.__exit__(exc, value, tb)
        self.loader.__exit__(exc, value, tb)

    @torch.inference_mode()
    def __call__(self):
        """
        Run evaluation
        """

        # TODO: create this dataloader
        for prompts, answers in self.loader:
            outputs = self.inference_engine.generate(prompts)

            # TODO correct this evaluation snippet

            loss = 0
            for output, answer in zip(outputs, answers):
                loss += 1 - int(output.endswith(f"{answer}."))
                print(output, answer, loss)

            bsz = len(prompts)
            scaling = bsz / self.loader.batch_size
            self.scaling += scaling

            self.loss += scaling * loss
            self.step += 1

            logger.debug(f"Evaluation: partial step: {self.step} - loss: {round(self.loss, 4):>7}")

        # rescale loss and save it
        self.loss /= self.scaling
        with open(self.log_path, "a") as f:
            print(json.dumps({"loss": self.loss} | self.metadata), file=f, flush=True)

        # remove temporary file
        self.tmp_file.unlink()
        return False



@torch.no_grad()
def run_evaluation(config: OnlineEvaluationConfig, model: BlockLanguageModel, step: int) -> None:
    computer = Evaluator(config, model, step=step)
    computer()

    if is_master_process():
        logger.info(f"Test loss: {round(computer.loss, 4):>7}")

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

    print(text_config)

    config = initialize_nested_object(OnlineEvaluationConfig, text_config, inplace=False)

    checkpoint_dir = Path("/private/home/vivc/memory/checkpoints/0/0000010000")

    # Model configuration
    # reload model parameters from checkpoint
    # checkpointer = Checkpointer()
    with open(f"{checkpoint_dir}/params.json") as f:
        model_config = {"model": json.load(f)}

    # # TODO: change some parameters (seq_len, flex_attention)

    # build model architecture
    model_config = build_config_with_model_dispatch(None, model_config)
    model: BlockLanguageModel = model_config.model_gen(model_config.model)

    # # TODO: parallelize model eventually
    model = model.to("cuda")

    # # load weights from checkpoint
    # # with checkpointer:
    state_dict = {"model": model.state_dict()}
    dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_dir)
    model.load_state_dict(state_dict["model"])

    evaluator = Evaluator(model, config, step=0)
    with evaluator:
        evaluator()

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------
# Evaluation Run
# ------------------------------------------------------------------------------


# # @dataclass
# # class EvaluationConfig:
#     cluster: ClusterConfig = field(default_factory=ClusterConfig)
#     orchestration: ...

#     evaluation: OnlineEvaluationConfig = field(default_factory=OnlineEvaluationConfig)


# @torch.no_grad()
# def eval(config: EvaluationConfig) -> None:
#     with ExitStack() as context_stack:
#         # ---------------------------------------------------------------------
#         # Handle preemption
#         # ---------------------------------------------------------------------

#         preemption: PreemptionHandler = context_stack.enter_context(PreemptionHandler())

#         # ---------------------------------------------------------------------
#         # Computing Environment
#         # ---------------------------------------------------------------------

#         cluster: ClusterManager = context_stack.enter_context(ClusterManager(config.cluster))

#         # ---------------------------------------------------------------------
#         # Instanciate logging
#         # ---------------------------------------------------------------------

#         context_stack.enter_context(Logger(config.orchestration.logging))

#         # ---------------------------------------------------------------------
#         # Build and Parallelize model
#         # ---------------------------------------------------------------------

#         logger.info("Building model")
#         model = Transformer(config.model)
#         model = cluster.initialize_model(model)

#         # ---------------------------------------------------------------------
#         # Recover Checkpoint
#         # ---------------------------------------------------------------------

#         # alias
#         train_step = config.orchestration.train_step
#         context_stack.enter_context(EvalCheckpointer(model, config.orchestration.checkpoint_path, train_step))

#         # ---------------------------------------------------------------------
#         # Run evaluation into chunks
#         # ---------------------------------------------------------------------

#         computer: EvalComputer = context_stack.enter_context(EvalComputer(config, model, train_step))

#         while next(computer):
#             # -----------------------------------------------------------------
#             # Handle preemption
#             # -----------------------------------------------------------------

#             if preemption():
#                 logger.warning("Preemption flag set")
#                 break

#             logger.debug(f"Evaluation. step: {computer.step} - loss: {round(computer.loss, 4):>7}")

#         # wandb logging
#         wandb: WandbLogger = context_stack.enter_context(WandbLogger(config.orchestration.wandb, config))
#         wandb({"test_loss": computer.loss, "step": train_step})

#     if is_master_process():
#         logger.info(f"Test loss: {round(computer.loss, 4):>7}")

#     logger.info("Evaluation done.")


# ------------------------------------------------------------------------------
# Asynchronous Evaluation Launcher
# ------------------------------------------------------------------------------
