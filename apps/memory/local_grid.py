# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Script to launch a grid locally. If a Slurm cluster is available, we advise to use src/nanollama/launcher.py.

Copyright (c) 2025 by the authors
"""

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from nanollama.utils import build_with_type_check, flatten_config, unflatten_config

logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Configuration Class
# ------------------------------------------------------------------------------


@dataclass
class LauncherConfig:
    """
    Configuration to launch a job.

    ### Parameters
    - script: script to launch (relative to the root of the project).
    - name: name of the job.
    - log_dir: directory to store logs.
    - overwrite: whether to overwrite the log directory.
    - copy_code: whether to copy the code to the log directory.
    - torchrun: whether to use torchrun to launch the job.
    - python_env: python environment to use.
    - grid: grid configuration to launch multiple jobs.
    - local_launch: launch configuration to setup the number of gpus.
    """

    script: str

    name: str = "composition_default"

    log_dir: str = ""
    overwrite: bool = False
    copy_code: bool = True
    torchrun: bool = False
    python_env: str = "default"

    grid: dict[str, Any] = field(default_factory=dict)
    local_launch: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Check validity of arguments and fill in missing values.
        """
        for key in self.grid:
            if isinstance(self.grid[key], str):
                self.grid[key] = eval(self.grid[key])

        if not self.log_dir:
            self.log_dir = str(Path.home() / "logs" / self.name)
            logger.info(f"No logging directory specified, default to {self.log_dir}")
        else:
            self.log_dir = os.path.expandvars(self.log_dir)

        # Set number of gpus if torchrun is used
        if self.torchrun and ("nb_gpus" not in self.local_launch):
            self.local_launch["nb_gpus"] = 1

        # recover python environment from the job was launched.
        if self.python_env:
            if self.python_env == "default":
                self.python_env = subprocess.check_output("which python", shell=True).decode("ascii").strip()
            else:
                self.python_env = f"{self.python_env}/bin/python"
            assert os.path.isfile(self.python_env)


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------


def copy_dir(input_dir: str, output_dir: str) -> None:
    rsync_cmd = (
        "rsync -ar --copy-links "
        "--exclude .git/ "
        # cache
        "--exclude .ruff_cache "
        "--exclude '*.egg-info' "
        "--exclude '__pycache__' "
        "--exclude '.pytest_cache' "
        # documentation
        "--exclude '*.md' "
        # data
        "--exclude '*.jsonl' "
        "--exclude '*.txt' "
        "--exclude '*.j2' "
        # configuration and scripts
        "--exclude .gitignore "
        "--exclude .vscode "
        "--exclude '*.sh' "
        "--exclude '*.toml' "
        "--exclude '*.yaml' "
        # checkpoints and runs
        "--exclude logs/ "
        "--exclude savings/ "
        # personal files and folders
        "--exclude wandb/ "
        "--exclude 'core.*' "
        "--exclude '*.ipynb' "
        "--exclude 'tmp_*' "
        # tests
        "--exclude tests/ "
        f"{input_dir}/ {output_dir}"
    )

    result = subprocess.run(rsync_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error:", result.stderr)


# ------------------------------------------------------------------------------
# Grid job utilities
# ------------------------------------------------------------------------------


def get_configs_from_grid(config: dict[str, Any], grid_config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Get a set of configurations from a base configuration and a grid configuration.

    ### Parameters
    config: base configuration.
    grid_config: grid configuration to launch a grid job.

    ### Returns
    List of configurations.
    """

    # get grid configurations as a list of flatten configs
    flatten_grid = flatten_config(grid_config)
    keys, all_values = zip(*flatten_grid.items(), strict=False)
    all_configs = [dict(zip(keys, v, strict=False)) for v in product(*all_values)]

    # merge on flatten config for simplicity
    config = flatten_config(config)
    return [unflatten_config(config | new_config) for new_config in all_configs]


# ------------------------------------------------------------------------------
# Job Launcher
# ------------------------------------------------------------------------------


LAUNCHER_SCRIPT = """#!/bin/bash

# activate conda environment
eval "$({conda_exe} shell.bash hook)"
conda activate {conda_env_path}

{go_to_code_dir}
# launch the job
export OMP_NUM_THREADS=1
{run_command}
"""


def launch_job(config: LauncherConfig, file_config: dict[str, Any]) -> None:
    """
    Launch a job. If the user has access to a Slurm cluster,
    the launch_job function from src/nanollama/launcher.py might be preferred.

    ### Parameters
    config: configuration to launch the job.
    run_config: training configuration of the job.
    """
    # alias
    nb_gpus = config.local_launch["nb_gpus"]
    run_config = file_config["run_config"]

    # logging directory
    log_dir = Path(config.log_dir)
    if log_dir.exists() and config.overwrite:
        confirm = input(
            f"Are you sure you want to delete the directory '{log_dir}'? This action cannot be undone. (yes/no): "
        )
        if confirm.upper().startswith("Y"):
            shutil.rmtree(log_dir)
            logger.info(f"Directory '{log_dir}' has been deleted.")
        else:
            logger.info("Operation cancelled.")
            sys.exit(0)
    log_dir.mkdir(exist_ok=True, parents=True)

    # copy code
    if config.copy_code:
        code_dir = log_dir / "code"
        code_dir.mkdir(exist_ok=True)
        logger.info(f"Copying code to {code_dir}.")
        copy_dir(os.getcwd(), code_dir)
        go_to_code_dir = "# go to code directory\n"
        go_to_code_dir = f"cd {code_dir}\n"
        go_to_code_dir = f"export PYTHONPATH=$PYTHONPATH:{code_dir}\n"
    else:
        go_to_code_dir = ""

    # write configs
    config_dir = log_dir / "tasks"
    config_dir.mkdir(exist_ok=True)
    config_paths = []
    if config.grid:
        # handling potential grid run
        logger.info("Writing grid configurations.")
        all_configs = get_configs_from_grid(run_config, config.grid)
        for i, nested_config in enumerate(all_configs, start=1):
            # set task dir and checkpoint path
            config_path = config_dir / f"{i}.yaml"
            nested_config["orchestration"]["task_id"] = str(i)

            # set evaluation job to synchronous so as not to depend on Slurm
            logger.info("Set evaluation job to synchronous.")
            nested_config["evaluation"]["asynchronous"] = False

            file_config["run_config"] = nested_config
            with open(config_path, "w") as f:
                yaml.dump(file_config, f, default_flow_style=False)
            config_paths.append(config_path)
    else:
        config_path = config_dir / "0.yaml"
        with open(config_path, "w") as f:
            yaml.dump(file_config, f, default_flow_style=False)

        config_paths.append(config_path)

    # define proper conda environment
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    conda_env_path = str(Path(config.python_env).parent.parent)

    # Loop over tasks
    run_command = ""
    for config_path in config_paths:
        # define the run command
        if config.torchrun:
            option_flags = f"--nproc_per_node={nb_gpus}"
            run_command += f"torchrun {option_flags} -m {config.script} {config_path}"
            run_command += "\n"
        else:
            run_command += f"python -u -m {config.script} {config_path}"
            run_command += "\n"

    bash_command = LAUNCHER_SCRIPT.format(
        name=config.name,
        log_dir=log_dir,
        nb_gpus=nb_gpus,
        conda_exe=conda_exe,
        conda_env_path=conda_env_path,
        go_to_code_dir=go_to_code_dir,
        run_command=run_command,
    )

    run_path = log_dir / "run.sh"
    with open(run_path, "w") as f:
        f.write(bash_command)

    logger.info("Launching job with `bash` command.")
    os.system(f"bash {run_path}")


def main() -> None:
    """
    Launch a grid job from configuration file specified by cli argument.

    Usage: launch the following command from the root directory
    ```
    python -m apps.memory.lauch_grid apps/my_app/my_config.yaml
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
    args = parser.parse_args()
    path = os.path.expandvars(args.config)

    # obtain configuration from file
    with open(path) as f:
        file_config: dict[str, Any] = yaml.safe_load(f)

    # initialize configuration
    config = build_with_type_check(LauncherConfig, file_config["launcher"], inplace=False)

    # launch job
    launch_job(config, file_config)


if __name__ == "__main__":
    main()
