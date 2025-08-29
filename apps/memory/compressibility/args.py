"""
Configuration and argument parsing for memory training on people with dependent attributes.

@ 2025, Ambroise Odonnat
"""

import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from src.nanollama.data.text import DataConfig, SourceConfig
from src.nanollama.distributed import ClusterConfig
from src.nanollama.launcher import SlurmConfig
from src.nanollama.model import EmbeddingModelConfig
from src.nanollama.monitor import (
    LightProfilerConfig,
    LoggerConfig,
    OrchestratorConfig,
    ProfilerConfig,
    PytorchProfilerConfig,
    WandbConfig,
)
from src.nanollama.optim import (
    OptimizerConfig,
)
from src.nanollama.utils import build_with_type_check, flatten_config, unflatten_config

from ..args import EvalConfig, OnlineEvaluationConfig, heritage_grid_id, heritage_launch_config
from ..dataset.dependent_data.generate import DATA_DIR

logger = logging.getLogger("nanollama")

# ------------------------------------------------------------------------------
# Simple Data Configuration
# ------------------------------------------------------------------------------


@dataclass
class MemoryDataConfig(DataConfig):
    """
    Data configuration for text data loader

    ### Attributes
    - tokenizer: tokenizer configuration
    - source: corpus of text specification as a list of weighted sources
    """

    nb_data: int = 0
    key: Literal["qa", "qatool", "biographies"] = ""
    data_dir: str = str(DATA_DIR)
    save_dir: str = str(DATA_DIR)
    sources: list[SourceConfig] = field(default_factory=list)

    def post_init(self) -> None:
        assert self.nb_data, "Number of data must be specified"
        assert self.key, "Key must be specified"

        self.data_dir = os.path.expandvars(self.data_dir)
        self.save_dir = str(Path(os.path.expandvars(self.save_dir)) / f"{self.key}_{self.nb_data}")
        self.sources = [SourceConfig(path=self.save_dir, weight=1)]
        super().post_init()

        logger.info("Building dataset from configuration")
        subprocess.run(
            [
                "python",
                "-m",
                "apps.memory.dataset.dependent_data.generate",
                "build",
                "--n-data",
                str(self.nb_data),
                "--key",
                self.key,
                "--save-dir",
                self.save_dir,
                "--data-dir",
                self.data_dir,
            ]
        )


# ------------------------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    compile: bool = False
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    orchestration: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    data: MemoryDataConfig = field(default_factory=MemoryDataConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # to allow for various implementation, we do not type checked model and tokenizer here
    model: dict[str, Any] = field(default_factory=dict)
    tokenizer: dict[str, Any] = field(default_factory=dict)
    model_callback: callable = field(init=False, default=None)

    def __post_init__(self) -> None:
        """
        Check validity of arguments and fill in missing values.
        """
        # restriction for cpu run
        self.cluster.post_init()
        if torch.device(self.cluster.device).type == "cpu":
            assert self.optim.fused is False, "Fused Adam is not supported on CPU"
            assert self.orchestration.profiler.pytorch.active is False, "Profiler is not supported on CPU"

        # manual post initialization of all modules
        self.orchestration.post_init()
        self.data.post_init()
        self.optim.post_init()

        # callback when building model in the training script: add seq_len if not specified
        def model_config_callback(model_config: EmbeddingModelConfig) -> None:
            if hasattr(model_config.block, "seq_len") and not model_config.block.seq_len:
                model_config.block.seq_len = self.data.seq_len - 1

        self.model_callback = model_config_callback

        # only post init evaluation if online
        eval = self.evaluation
        if eval.period and not eval.asynchronous:
            OnlineEvaluationConfig.post_init(self.evaluation)


def heritage_eval_config(run_config: dict[str, Any], launcher: dict[str, Any]) -> None:
    """
    Heritage of configuration from run to evaluation config.

    ### Parameters
    - run_config: configuration to run this file.
    - launcher: meta configuration to orchestrate the launch of this run.

    ### Note
    The logic of this file is a bit hard to parse, feel free to suggest better solutions.
    """

    logger.info("Heritage from run_config to eval_config")
    eval_config = run_config.get("evaluation", {})
    if eval_config.get("period", 0) <= 0:
        run_config["evaluation"] = eval_config
        return

    # flatten configurations for easier access
    flat_config = flatten_config(run_config)
    eval_config = flatten_config(eval_config)

    # special inheritance
    # orchestration
    eval_config["orchestration.name"] = flat_config["orchestration.name"] + "_eval"
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    eval_config["orchestration.log_dir"] = str(Path(flat_config["orchestration.log_dir"]) / "evals" / task_id)

    # generic inheritance
    configs_keys = [
        (DataConfig, "data"),
    ]

    # deal with data configuration being defined in its post_init method
    tmp_config = build_with_type_check(MemoryDataConfig, run_config["data"], inplace=False)
    tmp_config.post_init()
    sources = tmp_config.sources
    flat_config |= {"data.sources": [asdict(s) for s in sources]}

    if eval_config.get("asynchronous", False):
        # hack to add slurm inheritance
        flat_config |= flatten_config({"slurm": launcher.pop("slurm", {})})

        configs_keys += [
            (SlurmConfig, "slurm"),
            (ClusterConfig, "cluster"),
            (dict, "tokenizer"),
            (LoggerConfig, "orchestration.logging"),
            (ProfilerConfig, "orchestration.profiler"),
            (PytorchProfilerConfig, "orchestration.profiler.pytorch"),
            (LightProfilerConfig, "orchestration.profiler.ligth"),
            (WandbConfig, "orchestration.logging.wandb"),
        ]

    # proceed with inheritance
    for config_cls, cls_key in configs_keys:
        if config_cls is dict:
            for keys in flat_config:
                if keys.startswith(cls_key) and keys not in eval_config:
                    eval_config[keys] = flat_config[keys]
        else:
            for key, finfo in config_cls.__dataclass_fields__.items():
                if not finfo.init:
                    continue
                flat_key = f"{cls_key}.{key}"
                if flat_key not in eval_config and flat_key in flat_config:
                    eval_config[flat_key] = flat_config[flat_key]

    # merge configuration
    run_config["evaluation"] = unflatten_config(eval_config)


def build_train_config(file_config: dict[str, Any]) -> TrainingConfig:
    """
    Build configuration from file configuration for training run.

    ### Parameters
    - file_config: configuration as a dictionary.

    ### Returns
    - config: configuration as a dataclass.
    """

    if "run_config" in file_config:
        run_config: dict[str, Any] = file_config.pop("run_config")
    else:
        run_config = file_config
    launcher: dict[str, Any] = file_config.pop("launcher", {})

    heritage_launch_config(run_config, launcher)
    heritage_eval_config(run_config, launcher)

    # grid id system to handle special grid cases
    grid_id = run_config.get("grid_id", None)
    if grid_id is not None:
        run_config = heritage_grid_id(run_config, grid_id)

    config = build_with_type_check(TrainingConfig, run_config)
    return config
