# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Logging Managor

@ 2025, Meta
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from traceback import format_exception
from types import TracebackType
from typing import Any, Literal

import torch

from ..distributed import get_hostname, get_rank, is_master_process
from .wandb import WandbConfig, WandbLogger

logger = getLogger("nanollama")


@dataclass
class LoggerConfig:
    """
    Logger Configuration (both for stdout and metrics).

    ### Attributes
    - period: number of updates between each checkpoint
    - level: logging level
    - wandb: configuration of the wandb logger
    - stdout_path: path to the stdout log directory
    - metric_path: path to the metrics log directory
    """
    period: int = 100
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    wandb: WandbConfig = field(default_factory=WandbConfig)
    stdout_path: str = ""
    metric_path: str = ""

    def post_init(self) -> None:
        assert self.stdout_path, "stdout_path was not set"
        assert self.metric_path, "metric_path was not set"
        self.level = self.level.upper()
        self.wandb.post_init()


# ------------------------------------------------------------------------------
# Logging Manager
# ------------------------------------------------------------------------------


class Logger:
    def __init__(self, config: LoggerConfig, run_config: dict[str, Any] = None, eval: bool = False) -> None:
        # metric file
        rank = get_rank()
        if eval:
            self.path = None
            self.metric = str(config.metric_path)
        else:
            self.path = Path(config.metric_path)
            self.path.mkdir(parents=True, exist_ok=True)
            self.metric = str(self.path / f"raw_{rank}.jsonl")

        # wandb looger
        self.wandb = WandbLogger(config.wandb, run_config=run_config)

        # stdout file
        path = Path(config.stdout_path)
        path.mkdir(parents=True, exist_ok=True)
        stdout_file = path / f"device_{rank}.log"

        # configure stdout
        # ...remove existing handler
        getLogger().handlers.clear()

        # ...initialize logging stream
        log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s")
        log_level = getattr(logging, config.level)
        logger.setLevel(log_level)
        handler = logging.FileHandler(stdout_file, "a")
        handler.setFormatter(log_format)
        logger.addHandler(handler)

        # ...log to console
        if is_master_process() and "SLURM_JOB_ID" not in os.environ:
            handler = logging.StreamHandler()
            handler.setFormatter(log_format)
            logger.addHandler(handler)
            logger.info(f"Logging to {path}")

        logger.info(f"Running on machine {get_hostname()}")

        # start timer
        self.start_time = time.time()

    def __enter__(self) -> "Logger":
        """
        Open logging files.
        """
        self.metric = open(self.metric, "a")
        self.wandb.__enter__()
        return self

    def __call__(self, metrics: dict[str, Any]) -> None:
        """
        Report metrics to file.
        """
        metrics |= {"ts": time.time() - self.start_time}
        print(json.dumps(metrics), file=self.metric, flush=True)
        logger.info({k: round(v, 5) for k, v in metrics.items()})
        self.wandb(metrics)

    def report_statistics(self, model: torch.nn.Module) -> None:
        """
        Report gobal statistics about the model.
        """
        if is_master_process():
            numel = sum([p.numel() for _, p in model.named_parameters()])
            with open(self.path / "info_model.jsonl", "a") as f:
                print(json.dumps({"model_params": numel}), file=f, flush=True)
            logger.info(f"Model has {numel} parameters.")

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """
        Close logging files. Log exceptions if any.
        """
        self.metric.close()
        self.wandb.__exit__(exc, value, tb)
        if exc is not None:
            logger.error(f"Exception: {value}")
            logger.info("".join(format_exception(exc, value, tb)))
