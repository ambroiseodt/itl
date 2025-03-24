# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Utils to download Hugging Face datasets and cast them as JSONL files.

To download a dataset, the pipeline is the following:
- download file from HuggingFace, typically in parquet format
- convert parquet files to JSONL files
- shuffle and split JSONL files

@ Meta, 2025

### NOTES
This is work in progress.
Some datasets are actually mixed of various sources.
For these datasets, it is important not to mix the various sources into one (e.g. smoltalk).
"""

import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

CACHE_DIR = Path.home() / ".cache" / "datasets"
TARGET_DIR = Path.home() / "datasets"
LOG_DIR = Path.home() / "log_data"
logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Raw download from Hugging Face
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
    - name: name of the dataset to download.
    - target_dir: local directory to download the raw dataset to.

    ### Attributes
    - repo_id: repository ID of the dataset, such that the dataset can be downloaded from
    `https://huggingface.co/datasets/{repo_id}`
    - allow_patterns: patterns to filter "files and versions" to download from
    `https://huggingface.co/datasets/{repo_id}/tree/main`
    """

    name: DatasetName
    target_dir: str = None

    # hugging face parameters
    repo_id: str = field(init=False)
    allow_patterns: str = field(init=False, default=None)

    def __post_init__(self):
        self.name = DatasetName(self.name.lower())
        if self.target_dir is None:
            self.target_dir = str(CACHE_DIR / self.name.value)

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
    - name: name of the dataset to download.
    - target_dir: local directory to download the raw dataset to.
    - nb_workers: number of workers to use for downloading the dataset.
    - max_retries: maximum number of retries after time-out error to download the dataset.
    - retry_delay: delay between retries to download the dataset
    """
    config = DownloadDatasetArgs(name=name, target_dir=target_dir)

    logger.info(f"Downloading dataset from {config.repo_id}...")
    attempt = 0
    while True:
        try:
            snapshot_download(
                config.repo_id,
                repo_type="dataset",
                local_dir=str(config.target_dir),
                allow_patterns=config.allow_patterns,
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
    logger.info(f"Dataset downloaded to {config.target_dir}")


# ------------------------------------------------------------------------------
# Convert parquet datasets into JSONL ones
# ------------------------------------------------------------------------------


def datatrove_parquet_to_jsonl(
    name: str,
    source_dir: str = None,
    target_dir: str = None,
    batch_size: int = 256,
    delete: bool = False,
    log_dir: str = None,
) -> None:
    """
    Convert a dataset from parquet to JSONL format using datatrove.

    ### Parameters
    - name: name of the dataset.
    - source_dir: source directory containing the dataset.
    - target_dir: target directory to save the JSONL dataset.
    - n_tasks: number of tasks to use for conversion.
    - delete: whether to delete the original files after conversion.
    - log_dir: directory to save the logs.
    """
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter

    # parse missing arguments
    if target_dir is None:
        target_dir = TARGET_DIR
    target_dir = str(Path(target_dir) / name)

    if source_dir is None:
        source_dir = str(CACHE_DIR / name)
    if log_dir is None:
        log_dir = str(LOG_DIR / name)

    pattern = "**/*.parquet"
    n_tasks = len(list(Path(source_dir).rglob(pattern)))

    logger.info(f"running local pipeline executor for parquet to jsonl conversion with ntasks = {n_tasks}")
    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                source_dir,
                batch_size=batch_size,
                doc_progress=True,
                file_progress=True,
                glob_pattern=pattern,
            ),
            JsonlWriter(
                target_dir,
                output_filename=name + ".chunk.${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=n_tasks,
        logging_dir=log_dir,
    )
    pipeline_exec.run()

    if delete:
        logger.info(f"Dataset converted to JSONL format and saved to {target_dir}, removing original files.")
        shutil.rmtree(source_dir)


def parquet_to_jsonl_filewise(source_file: Path, target_file: Path, batch_size: int) -> None:
    """
    Convert a parquet file to JSONL using native python (slow, yet customizable).

    ### Parameters
    - source_file: path to the parquet file
    - target_file: path to the jsonl file
    - batch_size: number of rows to read at once
    """
    target_file.touch()
    parquet_file = pq.ParquetFile(source_file)
    total_rows = parquet_file.metadata.num_rows
    with open(target_file, "w") as jsonl_file:
        i = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            i += 1
            for row in zip(*batch.columns):
                row_dict = {col.name: str(value) for col, value in zip(batch.schema, row)}
                json.dump(row_dict, jsonl_file)
                jsonl_file.write("\n")
            logger.info(f"{source_file.name}: {100 * (batch_size * i) / total_rows:.3f} % done.")


def parquet_to_jsonl(
    name: str,
    source_dir: str = None,
    target_dir: str = None,
    batch_size: int = 256,
    delete: bool = False,
) -> None:
    """
    Convert a parquet dataset to JSONL one using native python (slow).

    ### Parameters
    - name: name of the dataset.
    - source_dir: source directory containing the dataset.
    - target_dir: target directory to save the JSONL dataset.
    - n_tasks: number of tasks to use for conversion.
    - delete: whether to delete the original files after conversion.
    - log_dir: directory to save the logs.
    """
    # parse missing arguments
    if target_dir is None:
        target_dir = TARGET_DIR
    target_dir = str(Path(target_dir) / name)

    if source_dir is None:
        source_dir = str(CACHE_DIR / name)

    # create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # list all the parquet files
    all_files = list(Path(source_dir).rglob("**/*.parquet"))

    with ThreadPoolExecutor(max_workers=len(all_files)) as executor:
        futures: list[Future] = []
        for rank, source_file in enumerate(all_files):
            target_file = target_dir / f"chunk.{rank:05d}.jsonl"
            futures.append(executor.submit(parquet_to_jsonl_filewise, source_file, target_file, batch_size=batch_size))

        for future in futures:
            future.result()

    if delete:
        logger.info(f"Dataset converted to JSONL format and saved to {target_dir}, removing original files.")
        shutil.rmtree(source_dir)


# ------------------------------------------------------------------------------
# TODO: Shuffle and split a JSON dataset based on terashuf
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
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire(
        {
            "hf_download": download_from_hf,
            "convert_to_jsonl": datatrove_parquet_to_jsonl,
        }
    )
