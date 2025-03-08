# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Generic Orchestrator Configuration, used to define the structure of logging directories (checkpoints, logs, ...).

@ 2025, Meta
"""

import os
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any

from ..distributed import is_master_process
from .checkpoint import CheckpointConfig
from .logger import LoggerConfig
from .profiler import ProfilerConfig
from .utility import UtilityConfig
from .wandb import WandbConfig

logger = getLogger("nanollama")


@dataclass
class OrchestratorConfig:
    """
    Orchestrator of logging organization for training runs.

    ### Attributes
    - log_dir: path to the root directory of the logs
    - name: name of the experiment
    - checkpoint: configuration of the checkpoint manager
    - logging: configuration of the logger
    - profiler: configuration of the profiler
    - utils: configuration of utility functions
    - wandb: configuration of the wandb logger
    """

    log_dir: str
    name: str = "train"

    # submanagers
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggerConfig = field(default_factory=LoggerConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    utils: UtilityConfig = field(default_factory=UtilityConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """

        # logging directory
        self.log_dir = os.path.expandvars(self.log_dir)
        log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # add discriminative information if array job
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")

        # checkpoint directory
        self.checkpoint.path = str(log_dir / "checkpoints" / task_id)

        # profile directory
        self.profiler.path = str(log_dir / "metrics" / task_id)

        # logging related
        self.logging.stdout_path = str(log_dir / "logs" / task_id)
        self.logging.metric_path = str(log_dir / "metrics" / task_id)
        self.wandb.path = str(log_dir / "wandb" / task_id)
        self.wandb.name = self.name + (f"_{task_id}" if task_id else "")

        # keep a mapping of job_id to task_id
        if task_id and is_master_process():
            job_id = os.environ.get("SLURM_JOB_ID")
            path = log_dir / "stdout"
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "id_mapping", "a") as f:
                f.write(f"task {task_id}: {job_id}\n")

        # check validity of submodule
        for module in self.__dict__.values():
            if hasattr(module, "check_init"):
                module.check_init()


# ------------------------------------------------------------------------------
# Evaluation Orchestrator
# ------------------------------------------------------------------------------


@dataclass
class EvalOrchestratorConfig:
    """
    Orchestrator of logging organization for evaluation runs.

    ### Attributes
    - log_dir: path to the root directory of the logs
    - name: name of the experiment
    - checkpoint: configuration of the checkpoint manager
    - logging: configuration of the logger
    - profiler: configuration of the profiler
    - utils: configuration of utility functions
    - wandb: configuration of the wandb logger
    """

    log_dir: str = ""
    name: str = "eval"

    # submanagers
    logging: LoggerConfig = field(default_factory=LoggerConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def check_init(self) -> None:
        """
        Check validity of arguments and fill in missing values.
        """

        # logging directory
        assert self.log_dir, "log_dir must be set."
        self.log_dir = os.path.expandvars(self.log_dir)
        log_dir = Path(self.log_dir)

        # add discriminative information if array job
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")

        # wandb directory (single dir for any steps)
        self.wandb.path = str(log_dir.parent / "wandb")
        self.wandb.name = self.name

        # same logic as OrchestratorConfig
        self.profiler.path = str(log_dir / "metrics" / task_id)
        self.logging.stdout_path = str(log_dir / "logs" / task_id)

        # fake metric path to be modified by in the evaluation config post-init
        self.logging.metric_path = self.log_dir

        for module in self.__dict__.values():
            if hasattr(module, "check_init"):
                module.check_init()

        # create directory
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # keep a mapping of job_id to task_id
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        if task_id != "0" and is_master_process():
            job_id = os.environ.get("SLURM_JOB_ID")
            path = Path(self.log_dir) / "stdout"
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "id_mapping", "a") as f:
                f.write(f"task {task_id}: {job_id}\n")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionnary to reinitialize it.
        """
        return {
            "log_dir": self.log_dir,
            "name": self.name,
            "logging": self.logging.to_dict(),
            "profiler": self.profiler.to_dict(),
            "wandb": self.wandb.to_dict(),
        }
