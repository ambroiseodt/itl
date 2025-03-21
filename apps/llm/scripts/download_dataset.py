# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Utils to download Hugging Face datasets and cast them as JSONL files.

@ Meta, 2025

### NOTES
This is work in progress.
Some datasets are actually mixed of various sources.
For these datasets, it is important not to mix the various sources into one (e.g. smoltalk).
"""

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
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
# TODO: Convert dataset formats to JSONL
# ------------------------------------------------------------------------------


def parquet_to_jsonl(
    dataset: str, log_dir: str, source_dir: str, target_dir: str, pattern: str = "**/*.parquet", n_tasks: int = 64
) -> None:
    """
    Convert a dataset from parquet to JSONL format.

    ### Parameters
    - dataset: The name of the dataset.
    - log_dir: The directory to save the logs.
    - source_dir: The source directory containing the dataset.
    - target_dir: The target directory to save the JSONL dataset.
    - n_tasks: The number of tasks to use for conversion.
    """
    print(f"running local pipeline executor for parquet to jsonl conversion with ntasks = {n_tasks}")
    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                source_dir,
                batch_size=256,
                file_progress=True,
                glob_pattern=pattern,
            ),
            JsonlWriter(
                target_dir,
                output_filename=dataset + ".chunk.${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=n_tasks,
        logging_dir=log_dir,
    )
    pipeline_exec.run()


# ------------------------------------------------------------------------------
# Utilities to shuffle and split datasets based on terashuf
# ------------------------------------------------------------------------------


def run_command(command: str) -> None:
    """ "
    Run a shell command and print it to the console.
    """
    logger.info(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def setup_terashuf(bin_dir: Path) -> Path:
    """Set up terashuf binary if needed.

    ### Parameters
    - bin_dir: The directory to install terashuf.

    ### Reference
    https://github.com/alexandres/terashuf/blob/master/README.md
    """
    terashuf_executable = bin_dir / "terashuf"

    if terashuf_executable.exists():
        logger.info("terashuf executable already exists. Skipping setup.")
        return terashuf_executable

    logger.info("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {bin_dir}")
    run_command(f"make -C {bin_dir}")
    return terashuf_executable


def shuffle_dataset(
    terashuf_executable: Path,
    dataset: str,
    src_dir: Path,
    tgt_dir: Path,
    src_ext: str,
    tgt_ext: str,
    n_chunks: int,
    seed: int,
) -> None:
    """Shuffle dataset chunks using terashuf

    ### Parameters
    - terashuf_executable: The path to the terashuf executable.
    - dataset: The dataset name.
    - src_dir: The source directory containing the dataset.
    - tgt_dir: The target directory to save the shuffled dataset.
    - src_ext: The source file extension.
    - tgt_ext: The target file extension.
    - n_chunks: The number of chunks to split the dataset into.
    - seed: The seed for shuffling.
    """
    logger.info(f"terashuf executable: {terashuf_executable}: {terashuf_executable.exists()}")
    os.environ["SEED"] = str(seed)
    run_command(
        # f"ulimit -n 100000 && "
        f"find {src_dir!s} -type f -name '*{src_ext}' -print0 | xargs -0 cat | {terashuf_executable!s} | "
        f"split -n r/{n_chunks} -d --suffix-length 2 --additional-suffix {tgt_ext} - {tgt_dir!s}/{dataset}.chunk."
    )


# ------------------------------------------------------------------------------
# Cli-interface
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
