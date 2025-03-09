# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Wandb Logger

@ 2025, Meta
"""

import json
import os
import sys
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from types import TracebackType
from typing import Any

try:
    import wandb
except ImportError:
    print("WARNING: wandb not installed.")

from ..distributed import is_master_process

logger = getLogger("nanollama")


@dataclass
class WandbConfig:
    """
    Wandb Configuration

    ### Attributes
    - active: whether to use wandb or not
    - entity: wandb user name
    - project: wandb project name
    - name: wandb run name
    - path: path to a file storing existing wandb run id
    """
    active: bool = False
    entity: str = ""
    project: str = "composition"
    name: str = ""
    path: str = ""

    def post_init(self) -> None:
        if self.active:
            assert self.name, "name was not set"
            assert self.path, "path was not set"


class WandbLogger:
    def __init__(self, config: WandbConfig, run_config: dict[str, Any] = None):
        self.active = config.active and is_master_process()
        if not self.active:
            return

        # open wandb api
        os.environ["WANDB_DIR"] = config.path
        id_file = Path(config.path) / "wandb.id"
        id_file.parent.mkdir(parents=True, exist_ok=True)

        self.entity = config.entity
        self.project = config.project
        self.id_file = id_file
        self.name = config.name

        self.run_config = run_config

    def __enter__(self) -> "WandbLogger":
        if not self.active:
            return self

        # Read run id from id file if it exists
        if os.path.exists(self.id_file):
            resuming = True
            with open(self.id_file) as file:
                run_id = file.read().strip()
        else:
            resuming = False

        if resuming:
            # Check whether run is still alive
            api = wandb.Api()
            run_state = api.run(f"{self.entity}/{self.project}/{run_id}").state
            if run_state == "running":
                logger.warning(f"Run with ID: {run_id} is currently active and running.")
                sys.exit(1)

            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                id=run_id,
                resume="must",
            )
            logger.info(f"Resuming run with ID: {run_id}")

        else:
            # Starting a new run
            self.run = wandb.init(
                config=self.run_config,
                project=self.project,
                entity=self.entity,
                name=self.name,
            )
            logger.info(f"Starting new run with ID: {self.run.id}")

            # Save run id to id file
            with open(self.id_file, "w") as file:
                file.write(self.run.id)

        self.run_config = None

        return self

    def __call__(self, metrics: dict) -> None:
        """Report metrics to wanbd."""
        if not self.active:
            return
        assert "step" in metrics, f"metrics should contain a step key.\n{metrics=}"
        wandb.log(metrics, step=metrics["step"])

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """Close wandb api."""
        if not self.active:
            return
        wandb.finish(exit_code=bool(exc))


def jsonl_to_wandb(
    path: str,
    name: str = "composition_default",
    project: str = "composition",
    entity: str = "",
    config: dict[str, Any] = None,
) -> None:
    """
    Push metric saved locally to wandb for visualization purposes
    """

    wandb.init(
        name=name,
        project=project,
        entity=entity,
        config=config,
    )
    with open(os.path.expandvars(path)) as f:
        for line in f:
            data = json.loads(line)
            wandb.log(data, step=data["step"])

    wandb.finish()
