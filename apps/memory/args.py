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

import torch

from src.nanollama.data.text import DataConfig, SourceConfig
from src.nanollama.distributed import ClusterConfig
from src.nanollama.launcher import LauncherConfig, SlurmConfig
from src.nanollama.model import EmbeddingModelConfig
from src.nanollama.monitor import (
    Checkpointer,
    EvalCheckpointConfig,
    EvalOrchestratorConfig,
    LoggerConfig,
    OrchestratorConfig,
    ProfilerConfig,
    WandbConfig,
)
from src.nanollama.optim import (
    OptimizerConfig,
)
from src.nanollama.utils import build_with_type_check, flatten_config, unflatten_config

from .dataset.generate import DATA_DIR

logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Evaluation Configurations
# ------------------------------------------------------------------------------


@dataclass
class OnlineEvaluationConfig:
    """
    Configuration to launch an online evaluation during a training run.

    ### Parameters
    - db_path: path to SQL database to retrive facts.
    - data: test data configuration.
    - tokenizer: tokenizer configuration.
    - checkpoint: evaluation checkpointing configuration.
    """

    db_path: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: EvalCheckpointConfig = field(default_factory=EvalCheckpointConfig)

    def post_init(self) -> None:
        assert self.db_path, "db_path should be specified."
        self.db_path = os.path.expandvars(self.db_path)

        self.data.post_init()
        self.checkpoint.post_init()


@dataclass
class EvaluationConfig(OnlineEvaluationConfig):
    """
    Configuration to schedule an offline evaluation during a training run.

    ### Parameters
    - model_dir: path to the model directory.
    - log_path: path to store the evaluation logs.
    - checkpoint_flag: file acting as a flag to avoid deleting checkpoints before evaluation
    - metadata: metadata to add to the evaluation metrics.
    - cluster: computing configuration.
    - orchestration: monitoring configuration.
    """

    model_dir: str = ""
    log_path: str = ""
    checkpoint_flag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    tokenizer: dict[str, Any] = field(default_factory=dict)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    orchestration: EvalOrchestratorConfig = field(default_factory=EvalOrchestratorConfig)

    def post_init(self) -> None:
        assert self.log_path, "log_path should be specified."
        assert self.model_dir, "model_dir should be specified."
        self.log_path = os.path.expandvars(self.log_path)

        # configuration not managed by orchestrator due to inheritance issues
        self.orchestration.post_init()
        self.checkpoint.path = str(Path(self.orchestration.log_dir) / "checkpoint")
        if self.checkpoint_flag:
            self.checkpoint.flag = str(Path(self.model_dir) / self.checkpoint_flag)
        self.orchestration.logging.metric_path = self.log_path

        self.cluster.post_init()
        super().post_init()

    def __post_init__(self):
        self.post_init()


def build_eval_config(file_config: dict[str, Any]) -> EvaluationConfig:
    """
    Build configuration from file configuration for evaluation run.

    ### Parameters
    - file_config: configuration as a dictionary.

    ### Returns
    - config: configuration as a dataclass.
    """

    if "run_config" in file_config:
        run_config = file_config["run_config"]
    else:
        run_config = file_config

    # initialize configuration
    config = build_with_type_check(EvaluationConfig, run_config, inplace=False)
    return config


# ------------------------------------------------------------------------------
# Evaluation Configuration for Training runs
# ------------------------------------------------------------------------------


@dataclass
class EvalConfig(EvaluationConfig):
    """
    Evaluation configuration to add to the training configuration.

    ### Parameters
    - period: period of evaluation
    - asynchronous: whether to launch evaluation asynchronously
    - slurm: slurm configuration for asynchronous evaluation

    It equally inherits from both the online and offline evaluation configuration.
    """
    period: int = 0
    asynchronous: bool = False
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    def __post_init__(self):
        """disable automatic post init"""
        ...


def build_eval_launch_config(
    eval_config: EvalConfig, orch: OrchestratorConfig, checkpoint: Checkpointer, step: int
) -> tuple[LauncherConfig, dict[str, Any]]:
    eval_flag = "flag"
    checkpoint.update(eval_flag=eval_flag)
    step_id = checkpoint.folder_name.format(step)

    eval_orch = eval_config.orchestration

    # specify training step etc.
    eval_config.log_path = str(Path(orch.logging.metric_path) / "eval_0.jsonl")
    eval_config.model_dir = str(Path(orch.checkpoint.path) / step_id)
    eval_config.checkpoint_flag = eval_flag
    eval_config.metadata = {"step": step}
    eval_orch.log_dir = str(Path(eval_orch.log_dir) / step_id)

    # launcher config
    eval_config.slurm.post_init()
    launch_config = build_with_type_check(
        LauncherConfig,
        {
            "name": eval_orch.name,
            "log_dir": eval_orch.log_dir,
            "overwrite": False,
            "copy_code": False,
            "script": "apps.memory.eval",
            "slurm": asdict(eval_config.slurm),
        },
    )

    # check and format config
    eval_dict = asdict(eval_config)
    eval_dict.pop("period")
    eval_dict.pop("asynchronous")
    eval_dict.pop("slurm")
    run_config = {"run_config": eval_dict}

    # remove step from log_dir
    eval_orch.log_dir = str(Path(eval_orch.log_dir).parent)

    return launch_config, run_config


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

    n_data: int = 0
    key: Literal["qa", "qatool", "biographies"] = ""

    data_dir: str = str(DATA_DIR)
    save_dir: str = str(DATA_DIR)
    sources: list[SourceConfig] = field(default_factory=list)

    def post_init(self) -> None:
        assert self.n_data, "Number of data must be specified"
        assert self.key, "Key must be specified"

        self.data_dir = os.path.expandvars(self.data_dir)
        self.save_dir = str(Path(os.path.expandvars(self.save_dir)) / f"{self.key}_{self.n_data}")
        self.sources = [SourceConfig(path=self.save_dir, weight=1)]
        super().post_init()

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


# ------------------------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------------------------


@dataclass
class TrainingConfig:
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
            assert self.orchestration.profiler.active is False, "Profiler is not supported on CPU"

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
