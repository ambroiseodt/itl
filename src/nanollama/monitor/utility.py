# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Utility Manager taking care of:
- seed setting
- garbage collection

Copyright (c) 2025 by the authors
"""

import gc
from dataclasses import dataclass
from logging import getLogger
from types import TracebackType

import torch

logger = getLogger("nanollama")


@dataclass
class UtilityConfig:
    seed: int = 42  # reproducibility
    period: int = 1000  # garbage collection frequency


class UtilityManager:
    def __init__(self, config: UtilityConfig):
        self.period = config.period
        self.seed = config.seed
        self.step = 0

    def __enter__(self) -> "UtilityManager":
        # set seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        # disable garbage collection
        gc.disable()
        gc.collect()
        return self

    def __call__(self) -> None:
        """
        Running garbage collection periodically.
        """
        self.step += 1
        if self.period <= 0:
            return
        if self.step % self.period == 0:
            logger.info("garbage collection")
            gc.collect()

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        # enable garbage collection
        gc.enable()
        return
