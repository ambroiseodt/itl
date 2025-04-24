# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Finetuning script of pre-trained models with online generation of batch of data.

@ 2025, Meta
"""

import argparse
import logging
import os
from typing import Any

import yaml

from .args import TrainingConfig, build_train_config

logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Finetuning loop
# ------------------------------------------------------------------------------


def finetune(config: TrainingConfig) -> None:
    """
    [WIP] This function is a work in progress.

    Notes
    -----
    For collaborators: currently, one has access to data as JsonIterator objects on which a tokenizer
    can then be applied to create the dataloader to fine-tune a HF pretrained model.

    TODO
    -----
    - define FinetuningConfig and build_finetuning_config objects with:
        - model, tokenizer to use
        - training and eval data to use (inspire from training config like debug.yaml)
        - threshold for stopping condition on hellaswag
        - training/optim parameters
    - load pretrained model and tokenizer from HF
    - Load dataset (train & eval)
    - Setup HF Trainer class (checkpoints, optim, eval)
    - Deal with Hellaswag condition (check along training that perf. on Hellaswag is > threshold)
    """

    # ---------------------------------------------------------------------
    # Load pretrained model and tokenizer
    # ---------------------------------------------------------------------

    logger.info("Loading pretrained model")
    # TODO: load pretrained model from HF

    logger.info("Loading tokenizer")
    # TODO: load corresponding tokenizer. Below a toy example

    # ---------------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------------

    logger.info("Building dataset")
    # TODO: create training and eval dataset
    # Helper: see JSONLIterator class from src/nanollama/data/text.py

    # ---------------------------------------------------------------------
    # Initialize Trainer class
    # ---------------------------------------------------------------------

    logger.info("Initialize Trainer class")
    # TODO: setup HF Trainer class

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    # TODO: implement stopping condition with performance on Hellaswag

    logger.info("Finetuning done.")


def main() -> None:
    """
    Launch a finetuning job from configuration file specified by cli argument.

    Usage:
    ```
    python -m apps.my_app.finetune apps/my_app/configs/my_config.yaml
    ```
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="Launch a finetuning job from configuration file")
    parser.add_argument("config", type=str, help="Path to configuration file")
    path = parser.parse_args().config

    # obtain configuration from file
    with open(os.path.expandvars(path)) as f:
        file_config: dict[str, Any] = yaml.safe_load(f)

    # initialize configuration
    config = build_train_config(file_config)

    # Launch finetuning
    finetune(config)


if __name__ == "__main__":
    main()
