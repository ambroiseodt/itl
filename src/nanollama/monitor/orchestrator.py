"""
Generic Orchestrator Configuration

This file is useful to define the structure of logging directories (checkpoints, evaluations, logs, metrics, ...).

#### License
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import os
from dataclasses import asdict, dataclass, field
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
    log_dir: str = ""
    name: str = "composition_default"

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
        if not self.log_dir:
            log_dir = Path.home() / "logs" / self.name
            self.log_dir = str(log_dir)
            logger.info(f"No logging directory set. Setting it to {self.log_dir}")
        else:
            self.log_dir = os.path.expandvars(self.log_dir)
            log_dir = Path(self.log_dir)

        log_dir.mkdir(parents=True, exist_ok=True)

        # add discriminative information if array job
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")

        # checkpoint directory
        self.checkpoint.path = str(log_dir / "checkpoints" / task_id)

        # profile directory
        self.profiler.path = str(log_dir / "metrics" / task_id)

        # logging related
        self.logging.stdout_path = str(log_dir / "logs" / task_id)
        self.logging.metric_path = str(log_dir / "metrics" / task_id)
        self.wandb.path = str(log_dir / "wandb" / task_id)
        self.wandb.name = f"{self.name}_{task_id}"

        # keep a mapping of job_id to task_id
        if task_id != "0" and is_master_process():
            job_id = os.environ.get("SLURM_JOB_ID")
            path = log_dir / "stdout"
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "id_mapping", "a") as f:
                f.write(f"task {task_id}: {job_id}\n")

        # check validity of submodule
        for module in self.__dict__.values():
            if hasattr(module, "__check_init__"):
                module.__check_init__()


# ------------------------------------------------------------------------------
# Evaluation Orchestrator
# ------------------------------------------------------------------------------


@dataclass
class EvalOrchestratorConfig:
    name: str = "composition_default"
    log_dir: str = ""

    # submanagers
    # checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggerConfig = field(default_factory=LoggerConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    utils: UtilityConfig = field(default_factory=UtilityConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __check_init__(self) -> None:
        """
        Check validity of arguments and fill in missing values.
        """

        # logging directory
        if not self.log_dir:
            log_dir = Path.home() / "logs_evals" / self.name
            # log_dir: PosixPath = parent_dir / "evals" / str(self.task_id) / f"{self.train_step:010d}"
            self.log_dir = str(log_dir)
            logger.info(f"No logging directory set. Setting it to {self.log_dir}")
        else:
            self.log_dir = os.path.expandvars(self.log_dir)
            log_dir = Path(self.log_dir)

        # wandb directory (single dir for any steps)
        self.wandb.path = str(log_dir.parent / "wandb")
        self.wandb.name = self.name

        # same logic as OrchestratorConfig
        self.profiler.path = str(log_dir / "metrics")
        self.logging.stdout_path = str(log_dir / "logs")
        self.logging.metric_path = str(log_dir / "metrics")

        for module in self.__dict__.values():
            if hasattr(module, "__check_init__"):
                module.__check_init__()

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
            "utils": asdict(self.utils),
            "wandb": self.wandb.to_dict(),
        }
