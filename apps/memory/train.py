# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Training script with online generation of batch of data.

@ 2025, Meta
"""

import argparse
import logging
import os
from contextlib import ExitStack
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

from src.nanollama.data.loader import DataLoader
from src.nanollama.data.text import MultipleSourcesTokenGenerator
from src.nanollama.distributed import ClusterManager, clean_environment, is_master_process
from src.nanollama.launcher import launch_job
from src.nanollama.model import (
    EmbeddingModel,
)
from src.nanollama.model.transformer import build_pretrain_mask, pretrain
from src.nanollama.monitor import (
    Checkpointer,
    Logger,
    PreemptionHandler,
    Profiler,
    UtilityManager,
)
from src.nanollama.optim import (
    OptimizerState,
    build_optimizer,
    build_scheduler,
)

from .args import TrainingConfig, build_eval_launch_config, build_train_config
from .eval import run_evaluation

logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------------------


def loss_func(preds: Tensor, targets: Tensor) -> Tensor:
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

        # ---------------------------------------------------------------------
        # Build and Parallelize model, optimizer, scheduler
        # ---------------------------------------------------------------------

        logger.info("Building model")
        model: EmbeddingModel = config.model_gen(config.model)
        model = cluster.build_model(model)

        if config.cluster.compile_model:
            logger.info("Compiling pipeline")
            pretrain_logic = torch.compile(pretrain)
        else:
            pretrain_logic = pretrain
        attn_mask = build_pretrain_mask(config.data.seq_len - 1, model.device)

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

        # TODO flops calculation
        raw_model = cluster.root_model
        token_per_step = config.data.seq_len * config.data.batch_size * config.optim.grad_acc_steps
        profiler.report_statistics(model=raw_model, token_per_step=token_per_step)
        metric_logger.report_statistics(raw_model)

        # ---------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------

        # aliases
        eval_period = config.evaluation.period
        log_period = config.orchestration.logging.period

        while optim_state.step < config.optim.steps:
            # alias
            step = optim_state.step

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

            profiler.start_timer()

            batch = next(dataloader)
            if cluster.device.type != "cpu":
                batch = batch.pin_memory()
            batch = batch.to(device=cluster.device, non_blocking=True)

            # the batch includes a mask associated to tokens supposed to be produced by the LLM.
            batch, loss_mask = batch.chunk(2)
            X_batch = batch[:, :-1]
            loss_mask = loss_mask[:, :-1].to(bool)
            y_batch = batch[:, 1:]

            profiler.end_timer("data_io_time")

            # -----------------------------------------------------------------
            # Forward and backward pass
            # -----------------------------------------------------------------

            profiler.start_timer()

            # forward propagation
            loss = pretrain_logic(model, X_batch, y_batch, mask=attn_mask, loss_mask=loss_mask)

            # rescale when using gradient accumulation (backprop on mean, not sum)
            loss = loss / config.optim.grad_acc_steps

            # backward propagation
            loss.backward()

            # gradient accumulation
            if optim_state.acc_step != 0:
                continue

            # extract gradient norm if logging
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.clip)

            # optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optim_state.step += 1

            profiler.end_timer("model_time", sync=True)

            # -----------------------------------------------------------------
            # Call monitors for garbage collection, checkpointing...
            # -----------------------------------------------------------------

            profiler.start_timer()

            profiler()
            checkpoint()
            utils()

            profiler.end_timer("monitor_time")

            # -----------------------------------------------------------------
            # Log metrics
            # -----------------------------------------------------------------

            profiler.start_timer()

            if log_period > 0 and step % log_period == 0:
                metrics = {"loss": loss.item(), "step": step}
                # extra metrics
                lr = optimizer.param_groups[0]["lr"]
                grad_norm = grad_norm.item()
                # TODO: if TP probably we should do the following instead
                # (grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm).item()
                metrics |= {
                    "lr": lr,
                    "grad_norm": grad_norm,
                }
                metric_logger(metrics)

            profiler.end_timer("log_time")

            # -----------------------------------------------------------------
            # Evaluation
            # -----------------------------------------------------------------

            profiler.start_timer()

            if eval_period > 0 and step % eval_period == 0:
                model.eval()

                # run evaluation now
                if not config.evaluation.asynchronous:
                    metrics = run_evaluation(config.evaluation, model=model, preemption=preemption)
                    metrics |= {"step": step}
                    metric_logger(metrics)

                # launch evaluation job on slurm
                elif is_master_process():
                    logger.info("Launching evaluation")
                    launch_config, run_config = build_eval_launch_config(
                        config.evaluation, config.orchestration, checkpoint, step
                    )

                    # launch job without device binding
                    def launcher(cfg: Any, run_cfg: dict[str, Any]) -> None:
                        with clean_environment():
                            launch_job(cfg, run_cfg)

                    # wait for checkpoint to be saved to launch job
                    checkpoint.process.add_done_callback(
                        lambda fut, cfg=launch_config, run_cfg=run_config: launcher(cfg, run_cfg)
                    )

                else:
                    logger.info("Saving model for evaluation")
                    checkpoint.update()

            profiler.end_timer("eval_time")

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

    parser = argparse.ArgumentParser(description="Launch a training job from configuration file")
    parser.add_argument("config", type=str, help="Path to configuration file")
    path = parser.parse_args().config

    # obtain configuration from file
    with open(os.path.expandvars(path)) as f:
        file_config: dict[str, Any] = yaml.safe_load(f)

    # initialize configuration
    config = build_train_config(file_config)

    train(config)


if __name__ == "__main__":
    main()
