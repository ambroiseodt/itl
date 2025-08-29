# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Profiler

@ 2025, Ambroise Odonnat
"""

import json
import subprocess
import time
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from threading import Thread
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

    def __enter__(self) -> "PytorchProfiler":
        self.profiler.__enter__()
        self.active = True
        logger.info(f"Pytorch profiler active. Traces will be saved at {self.path}")
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """
        Log profiler traces if not already done
        """
        if self.active:
            try:
                self.profiler.export_chrome_trace(self.path)
                self.profiler.__exit__(exc, value, tb)
            except AttributeError as e:
                logger.warning("Could not save profiler traces")
                logger.warning(e)
            logger.info(f"Pytorch profiler traces saved to {self.path}")
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
    period: int = 0
    path: str = ""
    refresh_rate: float = 0.001

    def post_init(self) -> None:
        assert self.path, "path was not set"


class LightProfiler:
    """
    Minimal profiler.
    """

    def __init__(self, config: LightProfilerConfig):
        self.path = str(Path(config.path) / f"prof_{get_rank()}.jsonl")
        self.period = config.period
        self.step = 0

        # placeholder and alias
        self.times = {}
        self.time = 0

        # memory information
        self.device = None
        self.capacity = 1
        self.init_memory_info()

        # gpu utilization information
        self.utilization: dict[str, float] = None
        self.utilization_thread: Thread = None
        self.utilization_scaling = 0

        self.utilization_thread = Thread(target=self._utilization_average, args=(config.refresh_rate,), daemon=True)

    def __enter__(self) -> "LightProfiler":
        logger.info(f"Light profiler active. Traces will be saved at {self.path}")
        self.file = open(self.path, "a")
        self.utilization_thread.start()
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        self.utilization_thread.join(timeout=0.01)
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
        metrics = {"step": self.step} | self.times | self.get_memory_metrics() | self.get_utilization_info()

        print(json.dumps(metrics), file=self.file, flush=True)
        logger.info({k: round(v, 5) for k, v in metrics.items()})

    # ------------------------------------------------------------------------------
    # Memory Information
    # ------------------------------------------------------------------------------

    def init_memory_info(self) -> None:
        # device
        rank = get_local_rank()
        self.device = torch.device(rank)
        try:
            self.capacity = torch.cuda.get_device_properties(self.device).total_memory / 100  # divide for percentage
        except Exception as e:
            logger.warning("Could not get device properties")
            logger.warning(e)

    def get_memory_metrics(self) -> dict[str, float]:
        # memory information
        cuda_info = torch.cuda.memory_stats(self.device)
        mem = cuda_info["active_bytes.all.peak"]
        mem_reserved = cuda_info["reserved_bytes.all.peak"]

        # reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

        return {
            "mem_GiB": mem / (1024**3),
            "mem_reserved_GiB": mem_reserved / (1024**3),
            "mem_percentage": mem / self.capacity,
            "num_alloc_retries": cuda_info["num_alloc_retries"],
            "num_ooms": cuda_info["num_ooms"],
        }

    # ------------------------------------------------------------------------------
    # GPU Utilization
    # ------------------------------------------------------------------------------

    @staticmethod
    def _utilization_snapshot() -> None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
            )
            utilization = result.stdout.strip().split("\n")
            return {f"utilization (cuda:{rank})": int(val) for rank, val in enumerate(utilization)}
        except subprocess.CalledProcessError as e:
            logger.warning(f"An error occurred: {e.stderr}")
            return None

    def _utilization_average(self, refresh_period: float = 1) -> None:
        while self.utilization_thread.is_alive():
            utilization = self._utilization_snapshot()
            if utilization is None:
                continue
            if self.utilization is None:
                self.utilization = utilization
                self.utilization_scaling = 1
            else:
                self.utilization = {k: (v + utilization[k]) for k, v in self.utilization.items()}
                self.utilization_scaling += 1
            time.sleep(refresh_period)

    def get_utilization_info(self) -> dict[str, float]:
        # get current average
        if self.utilization is None:
            metrics = {}
        else:
            metrics = {k: v / self.utilization_scaling for k, v in self.utilization.items()}

        # restart utilization average
        self.utilization = None
        return metrics

    # --------------------------------------------------------------------------
    # Timing information
    # --------------------------------------------------------------------------

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
        self.light = LightProfiler(config.light) if config.light.period > 0 else MockProfiler()
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
        return self

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
