"""
Checkpoint Manager

#### License
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta

#### Notes
Checkpointing does not support `DCP` yet.
See https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html

Checkpointing is done `synchronously` in the main process, this may slow down the training process.
See https://pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html.
"""

import re
import shutil
from asyncio import Future
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path, PosixPath
from types import TracebackType

import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from ..distributed import get_rank, is_master_process
from .monitor import Monitor

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# Stateful object to wrap all stateful objects
# ------------------------------------------------------------------------------


class AppState(Stateful):
    """
    Application state tracking model, optimizer and data state for checkpointing

    ### Parameter
    - model: model
    - optimizer: optimizer
    """

    def __init__(self, model: nn.Module, optimizer: Optimizer = None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, dict]:
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {"model": model_state_dict, "optim": optimizer_state_dict}

    def load_state_dict(self, state_dict: dict[str, dict]) -> None:
        set_state_dict(
            self.model, self.optimizer, model_state_dict=state_dict["model"], optim_state_dict=state_dict["optim"]
        )


@dataclass
class CheckpointConfig:
    period: int = 0
    keep_only: int = 0
    path: str = field(init=False, default="")

    def __check_init__(self):
        """Check validity of arguments."""
        assert self.path, "path was not set"


class Checkpointer(Monitor):
    """
    Checkpoint manager

    ### Attributes
    - period: number of updates between each checkpoint
    - keep_only: number of checkpoints to keep
    - path: path to the checkpoint directory
    - saved: whether the latest model has been saved
    - stateful_objects: various objects to checkpoints
    - app_state: AppState to track model and optimizer

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
    ):
        super().__init__(config)

        self.period = config.period
        self.keep_only = config.keep_only
        self.path = Path(config.path)
        self.path.mkdir(parents=True, exist_ok=True)

        # Create alias for the objects to monitor.
        self.app_state = AppState(model=model, optimizer=optimizer)
        self.model = model
        self.optimizer = optimizer
        self.stateful_objects = stateful_objects if stateful_objects else {}

        self.device_rank = get_rank()
        self.saved_step = 0
        self.step = 0

        self.checkpoint_process: Future = None

    def __enter__(self) -> "Checkpointer":
        """Enter checkpoint context by loading the last checkpoint"""
        path = self.get_last_checkpoint_path(self.path)
        if path:
            self.load(path)
        return self

    def update(self, eval: bool = False) -> None:
        """
        Checkpoint model, optimizer, scheduler and training state

        ### Parameters
        - eval: Whether to save the checkpoint for evaluation
        """
        # do not checkpoint twice
        if self.saved_step == self.step:
            return

        path = self.path / self.folder_name.format(self.step)
        path.mkdir(parents=False, exist_ok=True)

        # add evaluation flag, if needed
        if eval:
            eval_flag = path / "eval"
            eval_flag.touch()

        self.save(path)

        self._cleaning()
        self.saved_step = self.step

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """Exit checkpoint context by saving checkpoint if needed"""
        self.update()

    def load(self, path: str) -> None:
        """
        Load checkpoint from path

        ### Parameters
        path: path to the checkpoint
        """

        logger.info(f"Loading checkpoint from {str(path)}.")
        state_dict = self.get_state_dict()
        dcp.load(state_dict=state_dict, checkpoint_id=path)

        logger.info("Loading model weights and optimizer state.")
        set_state_dict(
            self.model, self.optimizer, model_state_dict=state_dict["model"], optim_state_dict=state_dict["optim"]
        )

        for name, obj in self.stateful_objects.items():
            logger.info("Loading {name}.")
            obj.load_state_dict(state_dict[name])

    def save(self, path: str) -> None:
        """
        Save checkpoint to path

        ### Parameters
        path: path to save the checkpoint
        """

        if self.checkpoint_process is not None:
            logger.info("Waiting for previous checkpoint to complete.")
            self.checkpoint_process.result()

        logger.info(f"Saving checkpoint at step {self.step} to {str(path)}.")
        state_dict = self.get_state_dict()
        self.checkpoint_process = dcp.async_save(state_dict, checkpoint_id=path)

    @torch.no_grad()
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
        if self.keep_only <= 0 or not is_master_process():
            return
        all_checkpoints = self._list_checkpoints(self.path)
        all_checkpoints.sort(key=lambda p: self._get_key_step(p.name))
        for prefix in all_checkpoints[: -self.keep_only]:
            if not (prefix / "eval").exists():
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
# Evaluation logic
# ------------------------------------------------------------------------------


class EvalCheckpointer:
    """
    Evaluation Checkpoint Manager
    """
    def __init__(self, model: nn.Module, path: str, train_step: int = None) -> None:
        self.model = model
        if train_step is None:
            path = Path(path)
            self.save_dir = Path(Checkpointer.get_last_checkpoint_path(path))
        else:
            self.save_dir = Path(path) / Checkpointer.folder_name.format(train_step)

    def __enter__(self):
        """Enter checkpoint context by loading the last checkpoint"""
        logger.info(f"Loading model from: {str(self.save_dir)}")
        state_dict = torch.load(self.save_dir / "checkpoint.pth", weights_only=True)
        self.model.load_state_dict(state_dict["model"])

        logger.info(f"Loading model from {str(self.save_dir)}.")
        state_dict = {"model": self.model.state_dict()}
        dcp.load(state_dict=state_dict, checkpoint_id=self.save_dir)

        logger.info("Loading model weights.")
        set_state_dict(self.model, model_state_dict=state_dict["model"])

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """
        Exit checkpoint context by remove `eval` flag

        #### See Also
        Checkpointer.update(eval=True)
        """
        eval_flag = self.save_dir / "eval"
        if eval_flag.exists():
            eval_flag.unlink()
