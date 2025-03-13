# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Utilities to load raw data from training runs into pandas dataframes, or numpy arrays.

@ 2025, Meta
"""

import json
import os
from logging import getLogger
from pathlib import PosixPath
from typing import Any

import numpy as np
import yaml

from ..utils import flatten_config

try:
    import pandas as pd
except ImportError:
    print("Pandas is not installed. Please install it using `pip install pandas`")

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# JSONL Loading Utilities
# ------------------------------------------------------------------------------


def get_jsonl_keys(path: str, readall: bool = True) -> list[str]:
    """
    Get keys from a jsonl file

    ### Parameters
    - path: path to the jsonl file
    - readall: wether to read all lines of the file or the first one only

    ### Returns
    keys: list of keys in the jsonl file
    """
    keys = set()
    with open(os.path.expandvars(path)) as f:
        for lineno, line in enumerate(f, start=1):
            try:
                keys |= json.loads(line).keys()
            except json.JSONDecodeError as e:
                logger.warning(f"Error reading line {lineno}: {e}")
            if not readall:
                break
    return list(keys)


def load_jsonl_to_numpy(path: str, keys: list[str] = None) -> dict[str, np.ndarray]:
    """
    Convert a jsonl file to a dictionnary of numpy array

    ### Parameters
    - path: path to the jsonl file
    - keys: list of keys to extract from the jsonl file

    ### Returns
    - a dictionnary of numpy arrays containing the data from the jsonl file
    """
    if keys is None:
        keys = get_jsonl_keys(path, readall=True)

    data: dict[str, list] = {key: [] for key in keys}
    with open(os.path.expandvars(path)) as f:
        # read jsonl as a csv with missing values
        for lineno, line in enumerate(f, start=1):
            try:
                values: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Error reading line {lineno}: {e}")
            for key in keys:
                data[key].append(values.get(key, None))
    return {k: np.array(v) for k, v in data.items()}


def load_jsonl_to_pandas(path: str, keys: list[str] = None) -> pd.DataFrame:
    """
    Convert a jsonl file to a pandas DataFrame

    ### Parameters
    - path: path to the jsonl file
    - keys: list of keys to extract from the jsonl file

    ### Returns
    - a dataframe containing the data from the jsonl file
    """
    if keys is None:
        keys = get_jsonl_keys(path, readall=True)

    data = load_jsonl_to_numpy(path, keys)
    return pd.DataFrame(data)


def load_results(metric_dir: PosixPath, file_prefix: str, keys: list[str] = None, merge: bool = True) -> pd.DataFrame:
    """
    Load multiple JSON files into a single pandas DataFrame.

    ### Parameters
    - metric_dir: path to metric logging directory.
    - file_prefix: identifier of the metric files to load.
    - keys: list of keys to extract from the jsonl files
    - merge: whether to merge the results into a single dataframe

    ### Returns
    - dataframe averaging results provided by various workers.
    """

    data = []
    for path in metric_dir.rglob(f"{file_prefix}_*.jsonl"):
        try:
            data.append(load_jsonl_to_pandas(path, keys=keys))
        except Exception as e:
            print(f"Error loading file {path}: {str(e)}")
            print(e)

    out = pd.concat(data)
    if "step" in out.columns and merge:
        out = out.groupby("step").mean().reset_index()
    return out


# ------------------------------------------------------------------------------
# Config Loading Utilities
# ------------------------------------------------------------------------------


def get_task_ids(log_dir: PosixPath) -> list[int]:
    """
    Get the task ids from the given log directory.

    ### Parameters
    - log_dir: path to logging directory.

    ### Returns
    - list of task ids.
    """
    task_ids = [int(p.name) for p in (log_dir / "metrics").glob("*") if p.is_dir()]
    task_ids.sort()
    return task_ids


def get_config_info(
    log_dir: PosixPath,
    task_id: int,
    keys: list[str],
    num_keys: list[str] = None,
    copy_num: bool = False,
    num_params: bool = False,
) -> dict[str, Any]:
    """
    Extract configuration informations.

    ### Parameters
    - log_dir: path to logging directory.
    - task_id: id of the task to extract information from.
    - keys: configuration keys to extract.
    - keys_as_id: keys to extract as .
    - copy_num: whether to copy the original values of the numerical keys.

    ### Returns
    - dictionary containing the extracted information.
    """
    res = {}

    if num_keys is None:
        num_keys = []

    # configuration information
    config_path = log_dir / "tasks" / f"{task_id}.yaml"
    with open(os.path.expandvars(config_path)) as f:
        config = flatten_config(yaml.safe_load(f))
    for key in keys:
        res[key] = config[f"run_config.{key}"]
    for key in num_keys:
        val = config[f"run_config.{key}"]
        all_val = config[f"launcher.grid.{key}"]
        res[f"id:{key}"] = all_val.index(val)
        if copy_num:
            res[key] = val

    # number of parameters
    if num_params:
        metric_path = log_dir / "metrics" / str(task_id)
        filepath = metric_path / "info_model.jsonl"
        with open(os.path.expandvars(filepath)) as f:
            res["nb_params"] = json.loads(f.readline())["model_params"]
    return res
