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

### TODO
Switch from huggingface_hub.snapshot_download to git lfs
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

CACHE_DIR = Path("/checkpoint/amaia/explore/datasets/reasoning/parquet")
TARGET_DIR = Path("/checkpoint/amaia/explore/datasets/reasoning/processed")
# CACHE_DIR = Path.home() / ".cache" / "datasets"
# TARGET_DIR = Path.home() / "datasets"
LOG_DIR = Path.home() / "log_data"
logger = logging.getLogger("nanollama")


# ------------------------------------------------------------------------------
# Raw download from Hugging Face
# ------------------------------------------------------------------------------


class DatasetName(Enum):
    AIME = "aime"
    ALGEBRAIC_STACK = "algebraic-stack"
    APPS = "apps"
    AQUA = "aqua"
    ARXIV = "arxiv"
    CODE_CONTESTS = "code-contests"
    CODEFORCES = "codeforces"
    COQ_GYM = "coq-gym"
    DCLM = "dclm"
    DEEPSCALER = "deepscaler"
    DEEPSEEK_PROVER = "deepseek-prover"
    EURUS_RL = "eurus-rl"
    FINEMATH = "finemath"
    FINEMATH_BIG = "finemath-big"
    FINEWEB_EDU = "fineweb-edu"
    FINEWEB_EDU_BIG = "fineweb-edu-big"
    GLAIVE_DISTILL = "glaive-distill"
    IMO_STEPS = "imo-steps"
    ISABELLE_PREMISE = "isabelle-premise"
    LEAN_WORKBOOK = "lean-workbook"
    LEETCODE = "leetcode"
    LILA = "lila"
    MATH = "math"
    MATH_INSTRUCT = "math-instruct"
    MATH_PILE = "math-pile"
    MBPP = "mbpp"
    META_MATH = "meta-math"
    NATURAL_REASONING = "natural-reasoning"
    NEMOTRON = "nemotron"
    NUMINA = "numina"
    NUMINA_TOOL = "numina-tool"
    OLYMPIAD_BENCH = "olympiad-bench"
    OMNI_MATH = "omni-math"
    OPEN_MATH_INSTRUCT = "open-math-instruct"
    OPEN_R1 = "open-r1"
    OPEN_WEB_MATH = "open-web-math"
    PROOF_PILE_2 = "proof-pile-2"
    SMOLTALK = "smoltalk"
    STACK_2 = "stack-2"
    STACK_EDU = "stack-edu"
    STACK_EDU_PYTHON = "stack-edu-python"
    STILL = "still"
    STILL_LONG = "still-long"
    SWEBENCH = "swebench"
    TACO = "taco"


@dataclass
class DownloadDatasetArgs:
    """
    Utility class to download a dataset from the Hugging Face Hub.

    ### Parameters
    - name: name of the dataset to download.
    - target_dir: local directory to download the raw dataset to.

    ### Attributes
    - url: repository ID of the dataset, such that the dataset can be downloaded from
    `https://huggingface.co/datasets/{url}`
    - allow_patterns: patterns to filter "files and versions" to download from
    `https://huggingface.co/datasets/{url}/tree/main`
    - revision: branch to download from
    """

    name: DatasetName
    target_dir: str = None

    # hugging face parameters
    url: str = field(init=False)
    allow_patterns: str = field(init=False, default=None)
    revision: str = field(init=False, default=None)
    file_format: str = field(init=False, default=None)

    def __post_init__(self):
        self.name = DatasetName(self.name.lower())
        if self.target_dir is None:
            self.target_dir = str(CACHE_DIR / self.name.value)

        match self.name:
            case DatasetName.AIME:
                self.url = "di-zhang-fdu/AIME_1983_2024"
                self.revision = "refs/convert/parquet"
                self.file_format = "parquet"

            case DatasetName.ALGEBRAIC_STACK:
                self.url = "EleutherAI/proof-pile-2"
                self.allow_patterns = "algebraic-stack/*"
                self.file_format = "jsonl.zst"

            case DatasetName.APPS:
                self.url = "codeparrot/apps"
                self.revision = "refs/convert/parquet"
                self.file_format = "parquet"

            case DatasetName.AQUA:
                self.url = "deepmind/aqua_rat"
                self.allow_patterns = "raw/*"
                self.file_format = "parquet"

            case DatasetName.ARXIV:
                # corresponds to an ArXiv snapshot made by RedPajama in 2023
                self.url = "EleutherAI/proof-pile-2"
                self.allow_patterns = "arxiv/*"
                self.file_format = "TODO"

            case DatasetName.CODE_CONTESTS:
                self.url = "deepmind/code_contests"
                self.file_format = "parquet"

            case DatasetName.CODEFORCES:
                self.url = "MatrixStudio/Codeforces-Python-Submissions"
                self.file_format = "parquet"

            case DatasetName.COQ_GYM:
                self.url = "https://github.com/princeton-vl/CoqGym"
                raise NotImplementedError("Download from github is not supported yet")

            case DatasetName.DCLM:
                self.url = "mlfoundations/dclm-baseline-1.0"
                self.allow_patterns = "*.jsonl.zst"
                self.file_format = "jsonl.zst"

            case DatasetName.DEEPSCALER:
                self.url = "agentica-org/DeepScaleR-Preview-Dataset"
                self.file_format = "json"

            case DatasetName.DEEPSEEK_PROVER:
                self.url = "deepseek-ai/DeepSeek-Prover-V1"
                self.file_format = "parquet"

            case DatasetName.EURUS_RL:
                self.url = "PRIME-RL/Eurus-2-RL-Data"
                self.file_format = "parquet"

            case DatasetName.FINEMATH:
                self.url = "HuggingFaceTB/finemath"
                self.allow_patterns = "finemath-4plus/*"
                self.file_format = "parquet"

            case DatasetName.FINEMATH_BIG:
                self.url = "HuggingFaceTB/finemath"
                self.allow_patterns = "finemath-3plus/*"
                self.file_format = "parquet"

            case DatasetName.FINEWEB_EDU:
                self.url = "HuggingFaceFW/fineweb-edu"
                self.allow_patterns = "sample/10BT/*"
                self.file_format = "parquet"

            case DatasetName.FINEWEB_EDU_BIG:
                self.url = "HuggingFaceFW/fineweb-edu"
                self.file_format = "parquet"

            case DatasetName.GLAIVE_DISTILL:
                self.url = "glaiveai/reasoning-v1-20m"
                self.file_format = "parquet"

            case DatasetName.IMO_STEPS:
                self.url = "roozbeh-yz/IMO-Steps"
                self.file_format = "lean"

            case DatasetName.ISABELLE_PREMISE:
                self.url = "Simontwice/premise_selection_in_isabelle"
                self.file_format = "json"

            case DatasetName.LEAN_WORKBOOK:
                self.url = "internlm/Lean-Workbook"
                self.file_format = "parquet"

            case DatasetName.LEETCODE:
                self.url = "greengerong/leetcode"
                self.file_format = "jsonl"

            case DatasetName.LILA:
                self.url = "allenai/lila"
                self.revision = "refs/convert/parquet"
                self.file_format = "parquet"

            case DatasetName.MATH:
                self.url = "hendrycks/competition_math"
                raise NotImplementedError("This dataset is not available on Hugging Face. Need custom logic.")

            case DatasetName.MATH_INSTRUCT:
                self.url = "TIGER-Lab/MathInstruct"
                self.file_format = "json"

            case DatasetName.MATH_PILE:
                self.url = "GAIR/MathPile"
                self.file_format = "jsonl.gz"

            case DatasetName.MBPP:
                self.url = "google-research-datasets/mbpp"
                self.file_format = "parquet"

            case DatasetName.META_MATH:
                self.url = "meta-math/MetaMathQA"
                self.file_format = "jsonl"

            case DatasetName.NATURAL_REASONING:
                self.url = "facebook/natural_reasoning"
                self.file_format = "json"

            case DatasetName.NEMOTRON:
                self.url = "nvidia/Llama-Nemotron-Post-Training-Dataset-v1"
                self.file_format = "json"

            case DatasetName.NUMINA:
                self.url = "AI-MO/NuminaMath-1.5"
                self.file_format = "parquet"

            case DatasetName.NUMINA_TOOL:
                self.url = "AI-MO/NuminaMath-TIR"
                self.file_format = "parquet"

            case DatasetName.OLYMPIAD_BENCH:
                self.url = "Hothan/OlympiadBench"
                self.file_format = "TODO"

            case DatasetName.OMNI_MATH:
                self.url = "KbsdJames/Omni-MATH"
                self.file_format = "jsonl"

            case DatasetName.OPEN_MATH_INSTRUCT:
                self.url = "nvidia/OpenMathInstruct-2"
                self.file_format = "parquet"

            case DatasetName.OPEN_R1:
                self.url = "open-r1/OpenR1-Math-220k"
                self.pattern = "all/*"
                self.file_format = "parquet"

            case DatasetName.OPEN_WEB_MATH:
                self.url = "open-web-math/open-web-math"
                self.file_format = "parquet"

            case DatasetName.PROOF_PILE_2:
                self.url = "EleutherAI/proof-pile-2"
                self.file_format = "jsonl.zst"

            case DatasetName.SMOLTALK:
                self.url = "HuggingFaceTB/smoltalk"
                self.file_format = "parquet"

            case DatasetName.STACK_2:
                self.url = "bigcode/the-stack-v2"
                self.file_format = "pointer"
                logger.warning("This will not download the Stack v2, but a list of pointer to download it.")

            case DatasetName.STACK_EDU:
                self.url = "HuggingFaceTB/stack-edu"
                self.file_format = "parquet"

            case DatasetName.STACK_EDU_PYTHON:
                self.url = "HuggingFaceTB/stack-edu"
                self.allow_patterns = "Python/*"
                self.file_format = "parquet"

            case DatasetName.STILL:
                self.url = "RUC-AIBOX/STILL-3-Preview-RL-Data"
                self.file_format = "parquet"

            case DatasetName.STILL_LONG:
                self.url = "RUC-AIBOX/long_form_thought_data_5k"
                self.file_format = "parquet"

            case DatasetName.SWEBENCH:
                self.url = "princeton-nlp/SWE-bench"
                raise NotImplementedError("The HuggingFace dataset is imcomplete, need to download from GitHub.")

            case DatasetName.TACO:
                self.url = "BAAI/TACO"
                self.allow_patterns = "ALL/*"
                self.file_format = "parquet"

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

    logger.info(f"Downloading dataset from {config.url}...")
    attempt = 0
    while True:
        try:
            snapshot_download(
                config.url,
                repo_type="dataset",
                local_dir=str(config.target_dir),
                allow_patterns=config.allow_patterns,
                revision=config.revision,
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
