# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Training script with online generation of batch of data.

@ 2025, Meta
"""

import logging
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from src.nanollama.data.loader import DataLoader
from src.nanollama.data.text import MultipleSourcesTokenGenerator
from src.nanollama.distributed import ClusterManager, clean_environment, get_rank, is_master_process
from src.nanollama.launcher import LauncherConfig, launch_job
from src.nanollama.model import (
    BlockLanguageModel,
)
from src.nanollama.monitor import (
    Checkpointer,
    Logger,
    PreemptionHandler,
    Profiler,
    UtilityManager,
    WandbLogger,
)
from src.nanollama.optim import (
    OptimizerState,
    build_optimizer,
    build_scheduler,
)
from src.nanollama.utils import initialize_nested_object

from .args import TrainingConfig
from .eval import EvaluationConfig, run_evaluation

logger = logging.getLogger("nanollama")


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
                    eval_config.slurm.check_init()
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


def main() -> None:
    """
    Launch a training job from configuration file specified by cli argument.

    Usage:
    ```
    python -m apps.my_app.train apps/my_app/configs/my_config.yaml
    ```
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    from .args import parse_args
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
