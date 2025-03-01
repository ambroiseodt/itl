"""
Training script with online generation of batch of data.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import logging
import os
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ...nanollama.data.loader import DataLoader
from ...nanollama.data.text import DataConfig, MultipleSourcesTokenGenerator
from ...nanollama.distributed import ClusterConfig, ClusterManager, get_rank
from ...nanollama.model import Transformer, TransformerConfig
from ...nanollama.model import transformer as tf
from ...nanollama.monitor import (
    Checkpointer,
    Logger,
    OrchestratorConfig,
    PreemptionHandler,
    Profiler,
    UtilityManager,
    WandbLogger,
)
from ...nanollama.optim import (
    OptimizerConfig,
    OptimizerState,
    build_optimizer,
    build_scheduler,
)
from ...nanollama.utils import initialize_nested_object

tf.FLEX_ATTENTION = True
_logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Configuration Class
# ------------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    orchestration: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    model: TransformerConfig = field(default_factory=TransformerConfig)
    model_gen: callable = field(init=False, default=Transformer)

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

        preemption: PreemptionHandler = context_stack.enter_context(PreemptionHandler())
        cluster: ClusterManager = context_stack.enter_context(ClusterManager(config.cluster))
        logger: Logger = context_stack.enter_context(Logger(config.orchestration.logging))
        utils: UtilityManager = context_stack.enter_context(UtilityManager(config.orchestration.utils))
        wandb: WandbLogger = context_stack.enter_context(
            WandbLogger(config.orchestration.wandb, run_config=asdict(config))
        )

        # ---------------------------------------------------------------------
        # Build and Parallelize model, optimizer, scheduler
        # ---------------------------------------------------------------------

        _logger.info("Building model")
        model: nn.Module = config.model_gen(config.model)
        model = cluster.build_model(model)

        _logger.info("Building optimizer")
        optimizer = build_optimizer(model, config.optim)
        scheduler = build_scheduler(optimizer, config.optim)
        optim_state = OptimizerState(step=0, acc_step=0)
        _logger.info("Done building optimizer")

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

        # ---------------------------------------------------------------------
        # Global information
        # ---------------------------------------------------------------------

        profiler: Profiler = context_stack.enter_context(Profiler(config.orchestration.profiler, state=optim_state))

        # ---------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------

        print(model)

        model.train()

        # aliases
        # eval_period = config.evaluation.period

        while optim_state.step < config.optim.steps:
            # handle preemption
            if preemption():
                _logger.warning("Preemption flag set")
                break

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

            profiler()
            checkpoint()

            print(get_rank(), loss.item())
            wandb({"loss": loss.item(), "step": optim_state.step})

    _logger.info("Training done.")


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
    if "run_config" in file_config:
        run_config: dict[str, Any] = file_config.pop("run_config")
    else:
        run_config = file_config
    launcher: dict[str, Any] = file_config.pop("launcher", {})

    # casting logging directory to run_config
    if "orchestration" not in run_config:
        run_config["orchestration"] = {}
    for key in ["name", "log_dir"]:
        if key in launcher and key not in run_config["orchestration"]:
            run_config["orchestration"][key] = launcher[key]

    # configuration inheritance between training and evaluation
    run_config["slurm"] = launcher.pop("slurm", {})
    run_config.pop("slurm")

    # grid id system to handle multiple datasets
    grid_id = run_config.get("grid_id", 0)
    try:
        run_config["data"]["path"] = run_config["data"]["path"].replace("$GRIDID", str(grid_id))
        run_config["evaluation"]["data"]["path"] = run_config["evaluation"]["data"]["path"].replace(
            "$GRIDID", str(grid_id)
        )
    except KeyError:
        pass

    # initialize configuration
    config = initialize_nested_object(TrainingConfig, run_config)

    # launch job
    train(config)


if __name__ == "__main__":
    main()
