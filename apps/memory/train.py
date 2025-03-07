# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Training script with online generation of batch of data.

@ 2025, Meta
"""

import logging
import os
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from src.nanollama.data.loader import DataLoader
from src.nanollama.data.text import DataConfig, MultipleSourcesTokenGenerator
from src.nanollama.data.tokenizer import TokenizerConfig
from src.nanollama.distributed import ClusterConfig, ClusterManager, clean_environment, get_rank, is_master_process
from src.nanollama.launcher import LauncherConfig, SlurmConfig, launch_job
from src.nanollama.model import (
    BlockLanguageModel,
    BlockLanguageModelConfig,
    build_config_with_model_dispatch,
)
from src.nanollama.monitor import (
    Checkpointer,
    Logger,
    LoggerConfig,
    OrchestratorConfig,
    PreemptionHandler,
    Profiler,
    ProfilerConfig,
    UtilityManager,
    WandbConfig,
    WandbLogger,
)
from src.nanollama.optim import (
    OptimizerConfig,
    OptimizerState,
    build_optimizer,
    build_scheduler,
)
from src.nanollama.utils import flatten_config, initialize_nested_object, unflatten_config

from .eval import EvaluationConfig, OnlineEvaluationConfig, run_evaluation
from .prompt_loader import DataConfig as EvalDataConfig

logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Configuration Class
# ------------------------------------------------------------------------------


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

    data: DataConfig = field(default_factory=DataConfig)
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
        self.model.block.seq_len = self.data.seq_len - 1

        # manual post initialization of all modules
        for module in self.__dict__.values():
            if hasattr(module, "__check_init__"):
                module.__check_init__()


# ------------------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------------------


def loss_func(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = preds.size(-1)
    return F.cross_entropy(preds.reshape(-1, vocab_size), targets.reshape(-1))


def train(config: TrainingConfig) -> None:
    with ExitStack() as context_stack:
        # ---------------------------------------------------------------------
        # Handle preemption, computing environment, logging, and utils
        # ---------------------------------------------------------------------

        preemption = PreemptionHandler()
        context_stack.enter_context(preemption)

        cluster = ClusterManager(config.cluster)
        context_stack.enter_context(cluster)

        metric_logger = Logger(config.orchestration.logging)
        context_stack.enter_context(metric_logger)

        utils = UtilityManager(config.orchestration.utils)
        context_stack.enter_context(utils)

        wandb = WandbLogger(config.orchestration.wandb, run_config=asdict(config))
        context_stack.enter_context(wandb)

        # ---------------------------------------------------------------------
        # Build and Parallelize model, optimizer, scheduler
        # ---------------------------------------------------------------------

        logger.info("Building model")
        model: BlockLanguageModel = config.model_gen(config.model)
        model = cluster.build_model(model)

        logger.info("Building optimizer")
        optimizer = build_optimizer(model, config.optim)
        scheduler = build_scheduler(optimizer, config.optim)
        optim_state = OptimizerState(step=0, acc_step=0)
        logger.info("Done building optimizer")

        # ---------------------------------------------------------------------
        # DataLoader
        # ---------------------------------------------------------------------

        token_gen = MultipleSourcesTokenGenerator(config.data, cluster.dp_mesh)
        dataloader: DataLoader = context_stack.enter_context(DataLoader(config.data, token_gen))

        # ---------------------------------------------------------------------
        # Recover Checkpoint
        # ---------------------------------------------------------------------

        checkpoint: Checkpointer = context_stack.enter_context(
            Checkpointer(
                config.orchestration.checkpoint,
                model=model,
                optimizer=optimizer,
                stateful_objects={"scheduler": scheduler, "dataloader": dataloader, "state": optim_state},
            )
        )
        checkpoint.saved_step = checkpoint.step = optim_state.step
        model.config = asdict(config.model)

        # ---------------------------------------------------------------------
        # Global information
        # ---------------------------------------------------------------------

        profiler: Profiler = context_stack.enter_context(Profiler(config.orchestration.profiler, state=optim_state))

        # TODO add a profiling

        # ---------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------

        # aliases
        eval_period = config.evaluation.period
        log_period = config.orchestration.logging.period

        while optim_state.step < config.optim.steps:
            # handle preemption
            if preemption():
                logger.warning("Preemption flag set")
                break

            model.train()

            # accumulation step
            optim_state.acc_step += 1
            optim_state.acc_step = optim_state.acc_step % config.optim.grad_acc_steps

            # -----------------------------------------------------------------
            # Batch of data (with reproducibility information)
            # -----------------------------------------------------------------

            # profiler.start_timer()
            batch = next(dataloader)
            if cluster.device.type != "cpu":
                batch = batch.pin_memory()

            batch = batch.to(device=cluster.device, non_blocking=True)

            # get mask associated to which token is supposed to be produced by the LLM.
            batch, mask = batch.chunk(2)
            X_batch = batch[:, :-1]
            mask = mask[:, :-1].to(bool)
            y_batch = batch[:, 1:][mask]

            # -----------------------------------------------------------------
            # Forward and backward pass
            # -----------------------------------------------------------------

            # forward propagation
            preds = model(X_batch)

            # loss on the tokens that have to be generated by the LLM
            loss = loss_func(preds[mask], y_batch)

            # rescale when using gradient accumulation (backprop on mean, not sum)
            loss = loss / config.optim.grad_acc_steps

            # backward propagation
            loss.backward()

            # gradient accumulation
            if optim_state.acc_step != 0:
                continue

            # optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optim_state.step += 1

            # -----------------------------------------------------------------
            # Call monitors for garbage collection, checkpointing...
            # -----------------------------------------------------------------

            profiler()
            checkpoint()
            utils()

            # alias
            step = optim_state.step

            # -----------------------------------------------------------------
            # Log metrics
            # -----------------------------------------------------------------

            if log_period > 0 and step % log_period == 0:
                metrics = {"loss": loss.item(), "step": step}
                metric_logger(metrics)
                wandb(metrics)

            # -----------------------------------------------------------------
            # Evaluation
            # -----------------------------------------------------------------

            profiler.start_timer()

            if eval_period > 0 and step % eval_period == 0:
                model.eval()

                # run evaluation now
                if not config.evaluation.asynchronous:
                    run_evaluation(config.evaluation, model=model)

                # launch evaluation job on slurm
                elif is_master_process():
                    # checkpoint
                    eval_flag = "flag"
                    checkpoint.update(eval_flag=eval_flag)

                    # alias
                    eval_config = config.evaluation
                    orch = config.orchestration
                    eval_orch = config.evaluation.orchestration
                    step_id = checkpoint.folder_name.format(step)

                    # specify training step etc.
                    eval_config.log_path = str(Path(orch.logging.metric_path) / f"eval_{get_rank()}.jsonl")
                    eval_config.model_dir = str(Path(orch.checkpoint.path) / step_id)
                    eval_config.checkpoint_flag = eval_flag
                    eval_config.metadata = {"step": step}
                    eval_orch.log_dir = str(Path(eval_orch.log_dir) / step_id)

                    # launcher config
                    eval_config.slurm.__check_init__()
                    launch_config = initialize_nested_object(
                        LauncherConfig,
                        {
                            "name": eval_orch.name,
                            "log_dir": eval_orch.log_dir,
                            "overwrite": False,
                            "copy_code": False,
                            "script": "apps.memory.eval",
                            "slurm": config.evaluation.slurm.to_dict(),
                        },
                    )

                    # check and format config
                    EvaluationConfig.__post_init__(eval_config)
                    eval_dict = eval_config.to_dict()
                    eval_dict.pop("period")
                    eval_dict.pop("asynchronous")
                    eval_dict.pop("slurm")
                    run_config = {"run_config": eval_dict}

                    # launch job without device binding
                    with clean_environment():
                        # wait for checkpoint to be saved to launch job
                        checkpoint.process.add_done_callback(
                            lambda fut, cfg=launch_config, run_cfg=run_config: launch_job(cfg, run_cfg)
                        )

                    eval_orch.log_dir = str(Path(eval_orch.log_dir).parent)

    logger.info("Training done.")


# ------------------------------------------------------------------------------
# Configuration utilities
# ------------------------------------------------------------------------------


def heritage_config(run_config: dict[str, Any], launcher: dict[str, Any]) -> None:
    """
    Heritage of configuration from launcher to run_config, and from run to evaluation config.

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

    # heritage from training to evaluation
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
        (TokenizerConfig, "tokenizer", "data.tokenizer"),
    ]

    if eval_config.get("asynchronous"):
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

    heritage_config(run_config, launcher)

    # grid id system to handle special grid cases
    grid_id = run_config.get("grid_id", None)
    if grid_id is not None:
        run_config = heritage_grid_id(run_config, grid_id)

    config = build_config_with_model_dispatch(TrainingConfig, run_config)
    return config


# ------------------------------------------------------------------------------
# Main file
# ------------------------------------------------------------------------------


def main() -> None:
    """
    Launch a training job from configuration file specified by cli argument.

    Usage:
    ```
    python -m apps.my_app.train apps/my_app/configs/my_config.yaml
    ```
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # parse file configuration path
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("config", type=str, help="Path to configuration file")
    path = parser.parse_args().config

    # obtain configuration from file
    with open(os.path.expandvars(path)) as f:
        file_config: dict[str, Any] = yaml.safe_load(f)

    # initialize configuration
    config = build_config(file_config)

    # launch job
    train(config)

if __name__ == "__main__":
    main()
