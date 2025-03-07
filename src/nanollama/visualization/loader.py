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

try:
    import pandas as pd
except ImportError:
    print("Pandas is not installed. Please install it using `pip install pandas`")

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# Loading Utilities
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
