# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Optimization Managers

@ 2025, Meta
"""

import math
from dataclasses import dataclass
from functools import partial

from torch import nn
from torch.optim import AdamW, Optimizer, lr_scheduler

# ------------------------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    # total number of update steps
    steps: int
    max_steps: int = 0
    # number of gradient accumulation before update
    grad_acc_steps: int = 1

    # AdamW parameters
    lr: float = 3e-4
    weight_decay: float = 0.1
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.95
    fused: bool = True

    # gradient clipping
    clip: float = float("inf")

    # scheduler parameters
    scheduler: str = "cosine"
    warmup: int = 2000
    lr_min_ratio: float = 0.1

    def post_init(self) -> None:
        if not self.max_steps:
            self.max_steps = self.steps


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    """
    Build optimizer and Scheduler
    """
    return AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        eps=config.epsilon,
        fused=config.fused,
    )


@dataclass
class OptimizerState:
    # nb of steps taken by the optimizer
    step: int
    # nb of accumulation steps done since last optimizer step
    acc_step: int

    def state_dict(self) -> dict[str, int]:
        return {"step": self.step, "acc_step": self.acc_step}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]


# ------------------------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------------------------


def build_scheduler(optimizer: Optimizer, config: OptimizerConfig) -> lr_scheduler.LambdaLR:
    """
    Initialize the scheduler state
    """
    if config.scheduler == "cosine":
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            partial(
                lr_cosine,
                warmup=config.warmup,
                steps=config.max_steps,
                min_ratio=config.lr_min_ratio,
            ),
        )

    if config.scheduler == "constant":
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lambda step: 1.0,
        )
    return scheduler


def lr_cosine(
    step: int,
    warmup: int,
    steps: int,
    min_ratio: float,
) -> float:
    """
    Cosine learning rate scheduler with warmup
    """
    assert warmup != steps, "Warmup and steps should not be equal"
    if step < warmup:
        lr = float(step) / warmup
    elif step <= steps:
        s = float(step - warmup) / (steps - warmup)
        lr = min_ratio + 0.5 * (1 - min_ratio) * (math.cos(math.pi * s) + 1)
    else:
        lr = min_ratio
    return lr
