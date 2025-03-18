# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Checkpoint Manager

@ 2025, Meta
"""

import json
import os
import re
import shutil
from asyncio import Future
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path, PosixPath
from types import TracebackType

import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from ..distributed import get_rank, is_master_process

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# Checkpointing logic at training time
# ------------------------------------------------------------------------------


@dataclass
class CheckpointConfig:
    """
    Checkpoint Configuration

    ### Attributes
    - period: number of updates between each checkpoint
    - nb_kept: number of checkpoints to keep
    - path: path to the checkpoint directory (set automatically by the orchestrator)
    """

    period: int = 0
    nb_kept: int = 0
    path: str = ""

    def post_init(self) -> None:
        if self.period > 0:
            assert self.path, "path was not set"


class Checkpointer:
    """
    Checkpoint manager

    ### Attributes
    - period: number of updates between each checkpoint
    - nb_kept: number of checkpoints to keep
    - path: path to the checkpoint directory
    - saved: whether the latest model has been saved

    ### Params
    - config: configuration object
    - model: model to checkpoint
    - optimizer: optimizer to checkpoint
    - stateful_objects: various objects to checkpoint
    """

    folder_name = "{:010d}"
    re_folder = r"\d{10}"
    re_digits = re.compile(r"\d+")

    def __init__(
        self,
        config: CheckpointConfig,
        model: nn.Module,
        optimizer: Optimizer = None,
        stateful_objects: dict[str, Stateful] = None,
        model_config: dict = None,
    ):
        self.period = config.period
        self.nb_kept = config.nb_kept
        self.model_config = model_config
        self.path = Path(config.path)
        self.path.mkdir(parents=True, exist_ok=True)

        # Create alias for the objects to monitor.
        self.model = model
        self.optimizer = optimizer
        self.stateful_objects = stateful_objects if stateful_objects else {}

        self.device_rank = get_rank()
        self.saved_step = 0
        self.step = 0
        self.period = config.period

        self.process: Future = None

    def sync_step(self, step: int) -> None:
        """
        Sync the step with the given value
        """
        self.saved_step = self.step = step

    def __enter__(self) -> "Checkpointer":
        """Enter checkpoint context by loading the last checkpoint"""
        path = self.get_last_checkpoint_path(self.path)
        if path:
            self.load(path)
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """Exit checkpoint context by saving checkpoint if needed"""
        # save checkpoint on exiting if not done already
        if self.saved_step != self.step:
            self.update()

        if self.process is not None:
            logger.info("Waiting for final checkpoint to complete.")
            self.process.result()

    def __call__(self) -> None:
        """Call update function periodically."""
        self.step += 1
        if self.period <= 0:
            return
        if self.step % self.period == 0:
            self.update()

    def update(self, eval_flag: str = "") -> None:
        """
        Checkpoint model, optimizer, scheduler and training state.

        ### Parameters
        - eval: Whether to save the checkpoint for evaluation
        """
        path = self.path / self.folder_name.format(self.step)
        path.mkdir(parents=False, exist_ok=True)

        # add evaluation flag, if needed
        if eval_flag:
            (path / f"eval_{eval_flag}").touch()

        # do not checkpoint twice
        if self.saved_step == self.step:
            return

        self.save(path)

        self._cleaning()
        self.saved_step = self.step

    def load(self, path: str) -> None:
        """
        Load checkpoint from path

        ### Parameters
        - path: path to the checkpoint
        """

        logger.info(f"Loading checkpoint from {str(path)}.")
        state_dict = self.get_state_dict()
        dcp.load(state_dict=state_dict, checkpoint_id=path)

        logger.info("Loading model weights and optimizer state.")
        set_state_dict(
            self.model, self.optimizer, model_state_dict=state_dict["model"], optim_state_dict=state_dict["optim"]
        )

        for name, obj in self.stateful_objects.items():
            logger.info(f"Loading {name}.")
            obj.load_state_dict(state_dict[name])

    def save(self, path: str) -> None:
        """
        Save checkpoint to path

        ### Parameters
        - path: path to save the checkpoint
        """

        if self.process is not None:
            logger.info("Waiting for previous checkpoint to complete.")
            self.process.result()

        logger.info(f"Saving checkpoint at step {self.step} to {str(path)}.")
        state_dict = self.get_state_dict()
        self.process = dcp.async_save(state_dict, checkpoint_id=path)

        if self.model_config is not None and is_master_process():
            with open(path / "params.json", "w") as f:
                json.dump(self.model_config, f)

    def get_state_dict(self) -> dict[str, dict]:
        """Return state dict of all tracked stateful objects."""
        model_sd, optimizer_sd = get_state_dict(self.model, self.optimizer)
        state_dict = {"model": model_sd, "optim": optimizer_sd}
        state_dict |= {name: obj.state_dict() for name, obj in self.stateful_objects.items()}
        return state_dict

    @classmethod
    def get_last_checkpoint_path(cls, path: str) -> str:
        """
        Get last existing checkpoint
        """
        folders = cls._list_checkpoints(path)
        if folders:
            return max(folders, key=lambda p: cls._get_key_step(p.name))
        return ""

    def _cleaning(self) -> None:
        """
        Clean up old checkpoints
        """
        if self.nb_kept <= 0 or not is_master_process():
            return
        all_checkpoints = self._list_checkpoints(self.path)
        all_checkpoints.sort(key=lambda p: self._get_key_step(p.name))
        for prefix in all_checkpoints[: -self.nb_kept]:
            if not any(prefix.glob("eval_*")):
                logger.info(f"Removing: {str(prefix)}")
                shutil.rmtree(prefix)

    @classmethod
    def _list_checkpoints(cls, path: str) -> list[PosixPath]:
        """
        List all existing checkpoints
        """
        return [p for p in path.iterdir() if p.is_dir() and re.match(cls.re_folder, p.name)]

    @classmethod
    def _get_key_step(cls, name: str) -> int:
        return int(re.findall(cls.re_digits, name)[-1])


# ------------------------------------------------------------------------------
# Checkpointing logic at training time
# ------------------------------------------------------------------------------


@dataclass
class EvalCheckpointConfig:
    """
    Configuration for the evaluation checkpoint manager.

    ### Attributes
    - path: path to checkpoint partial results
    - flag: file acting as a flag to avoid deleting checkpoints before evaluation
    """

    path: str = "$HOME/.tmp_nanollama"
    flag: str = ""

    def post_init(self) -> None:
        self.path = os.path.expandvars(self.path)


class EvalCheckpointer:
    """
    Automatically save and load evaluation state to avoid repeated computation.

    ### Parameters
    - config: evaluation checkpoint configuration
    - eval_state: evaluation state
    """

    def __init__(self, config: EvalCheckpointConfig, eval_state: Stateful):
        self.path = Path(config.path) / f"tmp_{get_rank()}.json"
        self.flag = config.flag
        self.state = eval_state

    def __enter__(self):
        """enter runtime context by reloading partial computation"""
        if self.path.exists():
            with open(self.path) as f:
                state = json.load(f)
            self.state.load_state_dict(state)
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """enter runtime context by saving (partial) computation"""
        # if the context was exited anormally, save partial computaton
        check_dir = self.path.parent
        if exc is not None:
            check_dir.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f:
                print(json.dumps(self.state.state_dict()), file=f, flush=True)

        # otherwise, delete temporary file, and checkpoint flag
        else:
            self.path.unlink(missing_ok=True)
            if check_dir.exists() and not any(check_dir.iterdir()):
                check_dir.rmdir()
            if self.flag:
                Path(self.flag).unlink(missing_ok=True)
