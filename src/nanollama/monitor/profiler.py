# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Profiler

@ 2025, Meta
"""

import json
import time
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from types import TracebackType

import torch
import torch.profiler as profiler

from ..distributed import get_local_rank, get_rank
from ..optim import OptimizerState

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# Pytorch Profiler
# ------------------------------------------------------------------------------


@dataclass
class PytorchProfilerConfig:
    active: bool = False
    wait: int = 1
    steps: int = 1
    path: str = ""

    def post_init(self) -> None:
        assert self.path, "path was not set"


class PytorchProfiler:
    """
    Wrapper around Pytorch Profiler, highly detailed, yet heavy.
    """

    ACTIVITIES = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    def __init__(self, config: PytorchProfilerConfig):
        self.path = str(Path(config.path) / f"prof_{get_rank()}.pt.trace.json")
        self.profiler = profiler.profile(
            activities=self.ACTIVITIES,
            schedule=torch.profiler.schedule(
                skip_first=0,
                wait=max(config.wait - 1, 0),
                warmup=min(config.wait, 1),
                active=config.steps,
                repeat=1,
            ),
            on_trace_ready=self._trace_ready_callback,
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            with_flops=True,
        )
        self.active = False

    def __enter__(self):
        self.profiler.__enter__()
        self.active = True
        logger.info(f"Pytorch profiler active. Traces will be saved at {self.path}")

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """
        Log profiler traces if not already done
        """
        if self.active:
            try:
                self.profiler.export_chrome_trace(self.path)
            except AttributeError as e:
                logger.warning("Could not save profiler traces")
                logger.warning(e)
            logger.info(f"Pytorch profiler traces saved to {self.path}")
            self.profiler.__exit__(exc, value, tb)
            self.active = False

    def __call__(self) -> None:
        """
        Call step function when profiler is active
        """
        if self.active:
            self.profiler.step()

    def _trace_ready_callback(self, prof: profiler.profile) -> None:
        self.__exit__(None, None, None)


# ------------------------------------------------------------------------------
# Lighter and Simpler Profiler
# ------------------------------------------------------------------------------


@dataclass
class LightProfilerConfig:
    active: bool = True
    period: int = 1
    path: str = ""

    def post_init(self) -> None:
        assert self.path, "path was not set"


class LightProfiler:
    """
    Minimal profiler.
    """

    def __init__(self, config: LightProfilerConfig):
        self.path = str(Path(config.path) / f"prof_{get_rank()}.jsonl")
        self.period = config.period
        self.active = False

        # placeholder and alias
        self.times = {}
        self.time = 0

        # device
        rank = get_local_rank()
        self.device = torch.device(rank)
        try:
            self.capacity = torch.cuda.get_device_properties(self.device).total_memory / 100  # divide for percentage
        except Exception as e:
            logger.warning("Could not get device properties")
            logger.warning(e)
            self.capacity = 1

    def __enter__(self):
        logger.info(f"Light profiler active. Traces will be saved at {self.path}")
        self.file = open(self.path, "a")

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        self.file.close()
        logger.info(f"Light profiler traces saved to {self.path}")

    def __call__(self) -> None:
        self.step += 1
        if self.period <= 0:
            return
        if self.step % self.period == 0:
            self.update()

    def update(self) -> None:
        """
        Call update function when profiler is active
        """
        # log profiler traces
        cuda_info = torch.cuda.memory_stats(self.device)

        # memory information
        mem = cuda_info["active_bytes.all.peak"]
        mem_reserved = cuda_info["reserved_bytes.all.peak"]

        metrics = self.times | {
            "step": self.step,
            "mem_GiB": mem / (1024**3),
            "mem_reserved_GiB": mem_reserved / (1024**3),
            "mem_percentage": mem / self.capacity,
            "num_alloc_retries": cuda_info["num_alloc_retries"],
            "num_ooms": cuda_info["num_ooms"],
        }

        print(json.dumps(metrics), file=self.file, flush=True)
        logger.info({k: round(v, 5) for k, v in metrics.items()})

        torch.cuda.reset_peak_memory_stats()

    def start_timer(self) -> None:
        self.time = time.time()

    def end_timer(self, name: str, sync: bool = False) -> None:
        if sync:
            torch.cuda.synchronize(self.device)
        self.times[name] = time.time() - self.time


# ------------------------------------------------------------------------------
# Generic Wrapper
# ------------------------------------------------------------------------------

@dataclass
class ProfilerConfig:
    pytorch: PytorchProfilerConfig = field(default_factory=PytorchProfilerConfig)
    light: LightProfilerConfig = field(default_factory=LightProfilerConfig)
    path: str = ""

    def post_init(self) -> None:
        assert self.path, "path was not set"
        self.pytorch.path = self.path
        self.light.path = self.path

        self.pytorch.post_init()
        self.light.post_init()


class MockProfiler:
    def __init__(self): ...
    def __enter__(self): ...
    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType): ...
    def __call__(self): ...
    def start_timer(self) -> None: ...
    def end_timer(self, name: str, **kwargs) -> None: ...


class Profiler:
    """
    Profiler Context

    #### Note
    One can easily tweak the implementation run multiple profilers simultaneously.
    """

    def __init__(self, config: ProfilerConfig, state: OptimizerState = None):
        self.pytorch = PytorchProfiler(config.pytorch) if config.pytorch.active else MockProfiler()
        self.light = LightProfiler(config.light) if config.light.active else MockProfiler()
        if self.pytorch or self.light:
            Path(config.path).mkdir(parents=True, exist_ok=True)

    def sync_step(self, step: int) -> None:
        """
        Sync the step with the given value
        """
        self.light.step = step

    def __enter__(self) -> "Profiler":
        self.pytorch.__enter__()
        self.light.__enter__()

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        self.pytorch.__exit__(exc, value, tb)
        self.light.__exit__(exc, value, tb)

    def __call__(self) -> None:
        """
        Call profilers
        """
        self.pytorch()
        self.light()

    def start_timer(self) -> None:
        """
        Start a timer
        """
        self.light.start_timer()

    def end_timer(self, name: str, **kwargs) -> None:
        """
        End timer and report time
        """
        self.light.end_timer(name, **kwargs)
