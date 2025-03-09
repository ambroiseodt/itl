"""
Configuration and argument parsing for memory training.

@ 2025, Meta
"""

import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from src.nanollama.data.text import DataConfig, SourceConfig
from src.nanollama.data.tokenizer import TokenizerConfig
from src.nanollama.distributed import ClusterConfig
from src.nanollama.launcher import SlurmConfig
from src.nanollama.model import (
    BlockLanguageModel,
    BlockLanguageModelConfig,
    build_config_with_model_dispatch,
)
from src.nanollama.monitor import (
    LoggerConfig,
    OrchestratorConfig,
    ProfilerConfig,
    WandbConfig,
)
from src.nanollama.optim import (
    OptimizerConfig,
)
from src.nanollama.utils import flatten_config, initialize_nested_object, unflatten_config

from .dataset.generate import DATA_DIR
from .eval import EvaluationConfig, OnlineEvaluationConfig
from .prompt_loader import DataConfig as EvalDataConfig

logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Configuration Class
# ------------------------------------------------------------------------------


@dataclass
class MemoryDataConfig(DataConfig):
    """
    Data configuration for text data loader

    ### Attributes
    - tokenizer: tokenizer configuration
    - source: corpus of text specification as a list of weighted sources
    """

    n_data: int = 0
    key: Literal["qa", "qatool", "biographies"] = ""

    data_dir: str = str(DATA_DIR)
    save_dir: str = str(DATA_DIR)
    sources: list[SourceConfig] = field(init=False, default=None)

    def __post_init__(self):
        assert self.n_data, "Number of data must be specified"
        assert self.key, "Key must be specified"
        assert self.tokenizer, "Tokenizer must be specified"

        self.data_dir = os.path.expandvars(self.data_dir)
        self.save_dir = str(Path(os.path.expandvars(self.save_dir)) / f"{self.key}_{self.n_data}")
        self.sources = [SourceConfig(path=self.save_dir, weight=1)]
        super().__post_init__()

    def check_init(self) -> None:
        logger.info("Building dataset from configuration")
        subprocess.run(
            [
                "python",
                "-m",
                "apps.memory.dataset.generate",
                "build",
                "--n-data",
                str(self.n_data),
                "--key",
                self.key,
                "--save-dir",
                self.save_dir,
                "--data-dir",
                self.data_dir,
            ]
        )


@dataclass
class EvalConfig(EvaluationConfig):
    period: int = 0
    asynchronous: bool = False

    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    def __post_init__(self):
        if not self.asynchronous and self.period > 0:
            OnlineEvaluationConfig.__post_init__(self)


@dataclass
class TrainingConfig:
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    orchestration: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    data: MemoryDataConfig = field(default_factory=MemoryDataConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    model: BlockLanguageModelConfig = field(default_factory=BlockLanguageModelConfig)
    model_gen: callable = field(init=False, default=BlockLanguageModel)

    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        """
        Check validity of arguments and fill in missing values.
        """
        # restriction for cpu run
        if self.cluster.device.type == "cpu":
            assert self.optim.fused is False, "Fused Adam is not supported on CPU"
            assert self.orchestration.profiler.active is False, "Profiler is not supported on CPU"

        # fill in missing values
        if hasattr(self.model.block, "seq_len") and not self.model.block.seq_len:
            self.model.block.seq_len = self.data.seq_len - 1

        # manual post initialization of all modules
        for module in self.__dict__.values():
            if hasattr(module, "check_init"):
                module.check_init()


def heritage_launch_config(run_config: dict[str, Any], launcher: dict[str, Any]) -> None:
    """
    Heritage of configuration from launcher to run_config.

    ### Parameters
    - run_config: configuration to run this file.
    - launcher: meta configuration to orchestrate the launch of this run.
    """

    logger.info("Heritage from launcher to run_config")
    if "orchestration" not in run_config:
        run_config["orchestration"] = {}
    for key in ["name", "log_dir"]:
        if key in launcher and key not in run_config["orchestration"]:
            run_config["orchestration"][key] = launcher[key]


def heritage_eval_config(run_config: dict[str, Any], launcher: dict[str, Any]) -> None:
    """
    Heritage of configuration from run to evaluation config.

    ### Parameters
    - run_config: configuration to run this file.
    - launcher: meta configuration to orchestrate the launch of this run.
    """

    logger.info("Heritage from run_config to eval_config")
    eval_config = run_config.get("evaluation", {})
    if eval_config.get("period", 0) <= 0:
        run_config["evaluation"] = eval_config
        return

    # flatten configurations for easier access
    flat_config = flatten_config(run_config)
    # hack to add slurm inheritance
    flat_config |= flatten_config({"_slurm": launcher.pop("slurm", {})})
    eval_config = flatten_config(eval_config)

    # special inheritance
    # orchestration
    eval_config["orchestration.name"] = flat_config["orchestration.name"] + "_eval"
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    eval_config["orchestration.log_dir"] = str(Path(flat_config["orchestration.log_dir"]) / "evals" / task_id)

    # generic inheritance
    configs_keys = [
        (EvalDataConfig, "data", "data"),
        (SourceConfig, "data.sources", "data.sources"),
        (TokenizerConfig, "tokenizer", "data.tokenizer"),
    ]

    # deal with data configuration being defined in its post_init method
    sources = initialize_nested_object(MemoryDataConfig, run_config["data"], inplace=False).sources
    flat_config |= {"data.sources": [asdict(s) for s in sources]}
    flat_config |= {
        "data.tokenizer.special_tokens": run_config.get("data", {}).get("tokenizer", {}).get("special_tokens", {})
    }

    if eval_config.get("asynchronous", False):
        configs_keys += [
            (SlurmConfig, "slurm", "_slurm"),
            (ClusterConfig, "cluster", "cluster"),
            (LoggerConfig, "orchestration.logging", "orchestration.logging"),
            (ProfilerConfig, "orchestration.profiler", "orchestration.profiler"),
            (WandbConfig, "orchestration.wandb", "orchestration.wandb"),
        ]
        eval_config["launcher"] = flatten_config(launcher)

    for config_cls, cls_key, inherited_key in configs_keys:
        for key, finfo in config_cls.__dataclass_fields__.items():
            if not finfo.init:
                continue
            eval_key = f"{cls_key}.{key}"
            train_key = f"{inherited_key}.{key}"
            if eval_key not in eval_config and train_key in flat_config:
                eval_config[eval_key] = flat_config[train_key]

    # merge configuration
    run_config["evaluation"] = unflatten_config(eval_config)


def heritage_grid_id(run_config: dict[str, Any], grid_id: int) -> None:
    """
    Specify configuration according to a grid id specified for job array.

    In the config, one can specify `launch.grid.grid_id: [...]`.
    The launcher will distributed these id into the run configs.
    This function will instanciate any `$GRID_ID` in the configuration with the right id.

    ### Parameters
    - run_config: configuration to run this file.
    - grid_id: id of the grid.
    """
    flat_config = flatten_config(run_config)
    for key in flat_config:
        if isinstance(flat_config[key], str):
            flat_config[key] = flat_config[key].replace("$GRID_ID", str(grid_id))
    return unflatten_config(flat_config)


def build_config(file_config: dict[str, Any]) -> TrainingConfig:
    """
    Build configuration from file configuration.

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

    config = build_config_with_model_dispatch(TrainingConfig, run_config)
    return config


def parse_args() -> TrainingConfig:
    """
    Parse command line arguments and build configuration.

    Returns:
        TrainingConfig: The parsed and processed training configuration
    """
    import argparse

    parser = argparse.ArgumentParser(description="Launch a training job from configuration file")
    parser.add_argument("config", type=str, help="Path to configuration file")
    path = parser.parse_args().config

    # obtain configuration from file
    with open(os.path.expandvars(path)) as f:
        file_config: dict[str, Any] = yaml.safe_load(f)

    # initialize configuration
    return build_config(file_config)
