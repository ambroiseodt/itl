# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Monitoring utilities to monitor runs.

@ 2025, Meta
"""

import os
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

from ..distributed import is_master_process
from .checkpoint import CheckpointConfig, Checkpointer, EvalCheckpointConfig, EvalCheckpointer
from .logger import Logger, LoggerConfig
from .preemption import PreemptionHandler
from .profiler import LightProfilerConfig, Profiler, ProfilerConfig, PytorchProfilerConfig
from .utility import UtilityConfig, UtilityManager
from .wandb import WandbConfig, WandbLogger

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
    """

    log_dir: str = ""
    name: str = "train"

    # submanagers
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggerConfig = field(default_factory=LoggerConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    utils: UtilityConfig = field(default_factory=UtilityConfig)

    def post_init(self) -> None:
        assert self.log_dir, "log_dir should be specified."

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
        self.logging.wandb.path = str(log_dir / "wandb" / task_id)
        self.logging.wandb.name = self.name + (f"_{task_id}" if task_id else "")

        # keep a mapping of job_id to task_id
        if task_id and is_master_process():
            job_id = os.environ.get("SLURM_JOB_ID")
            path = log_dir / "stdout"
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "id_mapping", "a") as f:
                f.write(f"task {task_id}: {job_id}\n")

        # check validity of submodules
        self.checkpoint.post_init()
        self.logging.post_init()
        self.profiler.post_init()


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
    - logging: configuration of the logger
    - profiler: configuration of the profiler
    - utils: configuration of utility functions
    """

    log_dir: str = ""
    name: str = "eval"

    # submanagers
    logging: LoggerConfig = field(default_factory=LoggerConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    def post_init(self) -> None:
        assert self.log_dir, "log_dir should be specified."

        # logging directory
        self.log_dir = os.path.expandvars(self.log_dir)
        log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # add discriminative information if array job
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")

        # profile directory
        self.profiler.path = str(log_dir / "metrics" / task_id)

        # logging related
        self.logging.stdout_path = str(log_dir / "logs" / task_id)
        # ...fake metric path to be modified by in the evaluation config post-init
        self.logging.metric_path = str(log_dir / "metrics" / task_id / "TBD")
        # ...wandb directory (single dir for any steps)
        self.logging.wandb.path = str(log_dir.parents[1] / "wandb" / task_id)
        self.logging.wandb.name = self.name

        # create directory
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # keep a mapping of job_id to task_id
        if task_id and is_master_process():
            job_id = os.environ.get("SLURM_JOB_ID")
            path = Path(self.log_dir) / "stdout"
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "id_mapping", "a") as f:
                f.write(f"task {task_id}: {job_id}\n")

        # check validity of submodules
        self.logging.post_init()
        self.profiler.post_init()
