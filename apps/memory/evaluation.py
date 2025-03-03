# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Evaluation script.

@ 2025, Meta
"""

import json
import os
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from types import TracebackType

import torch
import torch.distributed.checkpoint as dcp  # noqa: F401
import yaml  # noqa: F401

from src.nanollama.data.loader import DataLoader  # noqa: F401
from src.nanollama.data.text import DataConfig, MultipleSourcesTokenGenerator  # noqa: F401
from src.nanollama.distributed import ClusterConfig, ClusterManager, get_rank, is_master_process  # noqa: F401
from src.nanollama.inference import QueuedBatchedInference
from src.nanollama.model import BlockLanguageModel, build_config_with_model_dispatch  # noqa: F401
from src.nanollama.utils import initialize_nested_object  # noqa: F401

logger = getLogger("nanollama")

# ------------------------------------------------------------------------------
# Configuration Classes
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

    tmp_file: str = ""

    def __post_init__(self):
        if not self.tmp_file:
            self.tmp_file = Path(self.log_path).parent / f".tmp_eval_{get_rank()}.jsonl"


# ------------------------------------------------------------------------------
# Online Evaluation
# ------------------------------------------------------------------------------


class EvalComputer:
    """
    Evaluation manager

    Running evaluation into chunks in order to handle job preemption.

    Usage:
    ```python
    with EvalComputer(*args) as computer:
        while next(computer):
            pass
    """

    def __init__(self, model: BlockLanguageModel, config: OnlineEvaluationConfig, **metadata):
        # metadata (e.g. training step)
        self.metadata = metadata

        # file paths
        self.log_path = Path(config.log_path)
        self.tmp_file = Path(config.tmp_file)

        # data loader
        tokenizer = None
        self.loader = None

        # inference engine
        self.inference_engine = QueuedBatchedInference(model, tokenizer, config.db_path)

    def __enter__(self) -> "EvalComputer":
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
        self.inference_engine.__enter__(exc, value, tb)
        self.loader.__exit__(exc, value, tb)

    @torch.inference_mode()
    def __next__(self):
        try:
            # TODO: create this dataloader
            prompts, answer = next(self.loader)
            outputs = self.inference_engine.generate(prompts)

            # TODO correct this evaluation snippet
            bsz = len(prompts)
            scaling = bsz / self.loader.bsz
            self.scaling += scaling
            loss = sum(outputs == answer)
            self.loss += scaling * loss

            logger.debug(f"Evaluation: partial step: {self.step} - loss: {round(self.loss, 4):>7}")

        except StopIteration:
            # rescale loss and save it
            self.loss /= self.scaling
            with open(self.log_path, "a") as f:
                print(json.dumps({"loss": self.loss} | self.metadata), file=f, flush=True)

            # remove temporary file
            self.tmp_file.unlink()
            return False

    def __call__(self):
        """
        Run evaluation
        """
        with self:
            while next(self):
                ...


@torch.no_grad()
def run_evaluation(config: OnlineEvaluationConfig, model: BlockLanguageModel, step: int) -> None:
    computer = EvalComputer(config, model, step=step)
    computer()

    if is_master_process():
        logger.info(f"Test loss: {round(computer.loss, 4):>7}")


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


# # ------------------------------------------------------------------------------
# # Asynchronous Evaluation Launcher
# # ------------------------------------------------------------------------------


# @dataclass
# class AsyncEvaluationConfig:
#     """
#     Configuration to launch an evaluation during a training run asynchronously (as a separate process).
#     """

#     slurm: ...


# def tmp():
#     x = ClusterManager()


# checkpoint_dir = Path("/private/home/vivc/memory/checkpoints/0/0000010000")
# DB_PATH = "/private/home/vivc/code/memory/apps/memory/dataset/people.db"


# # Model configuration
# # reload model parameters from checkpoint
# # checkpointer = Checkpointer()
# with open(f"{checkpoint_dir}/params.json") as f:
#     config = {"model": json.load(f)}


# # TODO: change some parameters (seq_len, flex_attention)

# # build model architecture
# config = build_config_with_model_dispatch(None, config)
# model: BlockLanguageModel = config.model_gen(config.model)

# # TODO: parallelize model eventually
# model = model.to("cuda")

# # load weights from checkpoint
# # with checkpointer:
# state_dict = {"model": model.state_dict()}
# dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_dir)
# model.load_state_dict(state_dict["model"])

# # TODO DATA configuration

# config = yaml.safe_load("""
# sources:
# - path: $HOME/code/memory/apps/memory/dataset/qatool.jsonl
#   weight: 50
# tokenizer:
#     name: byte
# padding: true
# batch_size: 8
# seq_len: 257
# asynchronous: false
# """)
# data_config = initialize_nested_object(DataConfig, config)
# token_gen = MultipleSourcesTokenGenerator(data_config)
# dataloader = DataLoader(data_config, token_gen)
# with dataloader:
#     batch = next(dataloader)

# data, mask = batch.chunk(2)
# prompts = []
# prefix_lens = mask.argmax(dim=1) + 1

# print("printing data")
# decode_func = token_gen.generators[0].tokenizer.tokenizer.decode
# for datum, dlen in zip(data, prefix_lens):
#     prompts.append(decode_func(datum[:dlen]))
#     print(prompts[-1], "\n")

# tokenizer = token_gen.generators[0].tokenizer

# # generation
# inference_engine = QueuedBatchedInference(model, tokenizer, DB_PATH)

# with inference_engine:
#     outputs = inference_engine.generate(prompts)

# for dialog in outputs:
#     print(dialog, "\n")

# # ------------------------------------------------------------------------------
# # Offline Evaluation Function
# # ------------------------------------------------------------------------------
