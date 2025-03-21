# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Utils to download Hugging Face datasets and cast them as JSONL files.

@ Meta, 2025
"""

import io
import logging
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_DIR = Path.home() / ".cache" / "datasets"
logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Raw download from Hugging Face
# ------------------------------------------------------------------------------


def download_raw_hf_dataset(
    repo_id: str,
    target_dir: str,
    allow_patterns: str = None,
    nb_workers: int = 8,
    max_retries: int = 5,
    retry_delay: int = 10,
) -> None:
    """
    Download dataset from hugging from the Hugging Face Hub.

    ### Parameters
    - repo_id: The repository ID of the dataset.
    - target_dir: The local directory to download the dataset to.
    - allow_patterns: The patterns to allow for downloading the dataset.
    - nb_workers: The number of workers to use for downloading the dataset.
    - max_retries: The maximum number of retries after time-out error to download the dataset.
    - retry_delay: The delay between retries to download the dataset.
    """
    logger.info(f"Downloading dataset from {repo_id}...")
    attempt = 0
    while True:
        try:
            tmp_stdout = io.StringIO()
            with redirect_stdout(tmp_stdout):
                snapshot_download(
                    repo_id,
                    repo_type="dataset",
                    local_dir=str(target_dir),
                    allow_patterns=allow_patterns,
                    max_workers=nb_workers,
                )

            break
        except Exception:
            if attempt < max_retries - 1:
                logger.warning(f"Timeout occurred. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                attempt += 1
            else:
                raise
    logger.info(f"Dataset downloaded to {target_dir}")


# ------------------------------------------------------------------------------
# Front-end to download datasets
# ------------------------------------------------------------------------------


class DatasetName(Enum):
    DCLM = "dclm"
    FINEMATH = "finemath"
    FINEMATH_BIG = "finemath-big"
    FINEWEB = "fineweb"
    FINEWEB_BIG = "fineweb-big"
    SMOLTALK = "smoltalk"


@dataclass
class DownloadDatasetArgs:
    """
    Utility class to download a dataset from the Hugging Face Hub.

    ### Parameters
    - name: The name of the dataset to download.
    - target_dir: The local directory to download the raw dataset to.

    ### Attributes
    - repo_id: The repository ID of the dataset, such that the dataset can be downloaded from
    `https://huggingface.co/datasets/{repo_id}`
    - allow_patterns: patterns to filter "files and versions" to download from
    `https://huggingface.co/datasets/{repo_id}/tree/main`
    """

    name: DatasetName
    target_dir: str

    # hugging face parameters
    repo_id: str = field(init=False)
    allow_patterns: str = field(init=False, default=None)

    def __post_init__(self):
        self.name = DatasetName(self.name.lower())
        match self.name:
            case DatasetName.DCLM:
                self.repo_id = "mlfoundations/dclm-baseline-1.0"
                self.allow_patterns = "*.jsonl.zst"
            case DatasetName.FINEMATH:
                self.repo_id = "HuggingFaceTB/finemath"
                self.allow_patterns = "finemath-4plus/*"
            case DatasetName.FINEMATH_BIG:
                self.repo_id = "HuggingFaceTB/finemath"
                self.allow_patterns = "finemath-3plus/*"
            case DatasetName.FINEWEB:
                self.repo_id = "HuggingFaceFW/fineweb-edu"
                self.allow_patterns = "sample/10BT/*"
            case DatasetName.FINEWEB_BIG:
                self.repo_id = "HuggingFaceFW/fineweb-edu"
                self.allow_patterns = None
            case DatasetName.SMOLTALK:
                self.repo_id = "HuggingFaceTB/smoltalk"
                self.allow_patterns = None
            case _:
                raise ValueError(f"Unknown dataset: {self.name}")


def download_from_hf(
    name: str,
    target_dir: str = None,
    nb_workers: int = 8,
    max_retries: int = float("inf"),
    retry_delay: int = 10,
) -> None:
    """
    Download a dataset from the Hugging Face Hub.

    ### Parameters
    - name: The name of the dataset to download.
    - target_dir: The local directory to download the raw dataset to.
    - nb_workers: The number of workers to use for downloading the dataset.
    - max_retries: The maximum number of retries after time-out error to download the dataset.
    - retry_delay: The delay between retries to download the dataset
    """
    if target_dir is None:
        target_dir = str(DEFAULT_DIR / name)
    config = DownloadDatasetArgs(name=name, target_dir=target_dir)
    download_raw_hf_dataset(
        config.repo_id,
        config.target_dir,
        config.allow_patterns,
        nb_workers=nb_workers,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


# ------------------------------------------------------------------------------
# TODO: Convert raw datasets into JSONL files
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire(
        {
            "hf_download": download_from_hf,
        }
    )
