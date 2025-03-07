import json
import os
from logging import getLogger
from pathlib import PosixPath
from typing import Any

import yaml

from ..utils import flatten_config
from .loader import load_jsonl_to_numpy

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# Configuration Utilities
# ------------------------------------------------------------------------------


def extract_config_info(
    log_dir: PosixPath, task_id: int, keys: list[str], num_keys: list[str], copy_num: bool = False
) -> dict[str, Any]:
    """
    Extract configuration informations.

    Parameters
    ----------
    log_dir:
        Path to logging directory.
    task_id:
        Id of the task to extract information from.
    keys:
        The configuration keys to extract.
    num_keys:
        The keys to extract as numbers.
    copy_num:
        Whether to copy the original values of the numerical keys.
    """
    res = {}

    # configuration information
    config_path = log_dir / "tasks" / f"{task_id}.yaml"
    with open(os.path.expandvars(config_path)) as f:
        config = flatten_config(yaml.safe_load(f))
    for key in keys:
        res[key] = config[f"run_config.{key}"]
    for key in num_keys:
        val = config[f"run_config.{key}"]
        all_val = config[f"launcher.grid.{key}"]
        res[f"num:{key}"] = all_val.index(val)
        if copy_num:
            res[key] = val

    # number of parameters
    metric_path = log_dir / "metrics" / str(task_id)
    filepath = metric_path / "info_model.jsonl"
    with open(os.path.expandvars(filepath)) as f:
        res["nb_params"] = json.loads(f.readline())["model_params"]
    return res


def get_losses(metric_path: PosixPath, steps: list, eval: bool = False) -> dict[str, float]:
    """
    Get the loss for the given metric path.

    Parameters
    ----------
    metric_path:
        Path to metric files.
    steps:
        Training steps to snapshot the loss.
    eval:
        Whether to consider the testing or training loss.

    Returns
    -------
    The loss for the given metric path.
    """
    res = {}
    prefix = "eval" if eval else "raw"

    # compute the loss
    loss = None
    world_size = 0
    for filepath in metric_path.glob(f"{prefix}_*.jsonl"):
        keys = ["loss", "step"]
        data = load_jsonl_to_numpy(filepath, keys=keys)
        if loss is None:
            loss = data["loss"]
        else:
            loss += data["loss"]
        world_size += 1
    logger.debug(f"Directory {metric_path} World_size: {world_size}")
    loss /= world_size

    # extract statistics
    step = data["step"]
    for snapshot in steps:
        idx = (step == snapshot).argmax()
        res[f"loss_{snapshot}"] = loss[idx].item()
    res["best"] = loss.min().item()
    return res


def read_indented_jsonl(filepath: str) -> list[dict]:
    data = []
    with open(filepath) as file:
        content = file.read()

    # split the content into individual JSON objects
    json_objects = content.split("}\n{")

    # adjust format
    if json_objects:
        json_objects[0] = json_objects[0] + "}"
        json_objects[-1] = "{" + json_objects[-1]
        for i in range(1, len(json_objects) - 1):
            json_objects[i] = "{" + json_objects[i] + "}"

    # parse each JSON object
    for json_str in json_objects:
        json_object = json.loads(json_str)
        data.append(json_object)
    return data


# ------------------------------------------------------------------------------
# Postprocessing Utilities
# ------------------------------------------------------------------------------


def get_task_ids(log_dir: PosixPath) -> list[int]:
    """
    Get the task ids from the given log directory.

    Parameters
    ----------
    log_dir:
        Path to logging directory.

    Returns
    -------
    The list of task ids.
    """
    task_ids = [int(p.name) for p in (log_dir / "metrics").glob("*") if p.is_dir()]
    task_ids.sort()
    return task_ids


def process_results(
    log_dir: PosixPath, keys: list[str], num_keys: list[str], steps: list[int], eval: bool, copy_num: bool = False
) -> None:
    """
    Process the results of the given experiments.

    Parameters
    ----------
    log_dir:
        Path to logging directory.
    keys:
        The configuration keys to extract.
    num_keys:
        The keys to extract as numbers.
    steps:
        Training steps to snapshot the loss.
    eval:
        Whether to consider the testing or training loss.
    copy_num:
        Whether to copy the original values of the numerical keys.
    """
    logger.info(f"Processing results in {log_dir}")
    all_task_ids = get_task_ids(log_dir)
    for task_id in all_task_ids:
        try:
            metric_path = log_dir / "metrics" / str(task_id)
            res = extract_config_info(log_dir, task_id, keys, num_keys, copy_num=copy_num)
            res |= get_losses(metric_path, steps, eval=eval)

            with open(os.path.expandvars(metric_path / "process.json"), "w") as f:
                print(json.dumps(res, indent=4), file=f, flush=True)
        except Exception as e:
            print(log_dir / "metrics" / str(task_id))
            logger.error(f"Error processing task {task_id}: {e}")
            continue
