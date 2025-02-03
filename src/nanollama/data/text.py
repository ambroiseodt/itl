"""
Text loader

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from numpy.random import default_rng
from torch.distributed.checkpoint.stateful import Stateful

from ..distributed import get_rank, get_world_size
from .tokenizer import TokenizerConfig
from .utils import generate_seeds

# ------------------------------------------------------------------------------
# Configuration classes
# ------------------------------------------------------------------------------


@dataclass
class SourceConfig:
    """
    Text source configuration

    Attributes
    ----------
    path: path to text files
    weight: weighting of the source in the datamix
    """

    path: str
    weight: float


@dataclass
class DataConfig:
    """
    Data configuration for text data loader

    Attributes
    ----------
    source: corpus of text specification as a list of weighted sources
    tokenizer: tokenizer configuration
    seq_len: sequence length
    batch_size: batch size
    asynchronous: whether to use asynchronous data loading
    buffer_size: number of batches to bufferize asynchronously for data loading
    """

    sources: list[SourceConfig]
    tokenizer: TokenizerConfig

    seq_len: int = 0
    batch_size: int = 0
    seed: int = 0
    asynchronous: bool = True
    buffer_size: int = 4

    def __post_init__(self):
        pass


# ------------------------------------------------------------------------------
# Stateful Information
# ------------------------------------------------------------------------------


@dataclass
class DataLoaderState(Stateful):
    source_choice_rng_state: dict[str, Any]
    sources_rng_state: list[dict[str, Any]]

    def state_dict(self) -> dict[str, Any]:
        return {
            "source_choice_rng_state": self.source_choice_rng_state,
            "sources_rng_state": self.sources_rng_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.source_choice_rng_state = state_dict["source_choice_rng_state"]
        self.sources_rng_state = state_dict["sources_rng_state"]


def build_dataloader_state(config: DataConfig) -> DataLoaderState:
    """
    Initialize the state of random number generators.

    TODO
    ----
    Here I generate only one seed per worker, it is easy to change the codebase to generate multiple seed
    """
    rank = get_rank()
    world_size = get_world_size()
    _, seeds = generate_seeds(0, 1, config.seed, rank, world_size)
    rng_state = default_rng(seeds[-1]).bit_generator.state
    return DataLoaderState(rng_state=rng_state)


# ------------------------------------------------------------------------------
# Text Generator
# ------------------------------------------------------------------------------


class SequenceGenerator:
    """
    Generate sequences iteratively in order to feed a tokenizer
    """

    def __init__(self, config: DataConfig):
        # all sources

        # weights for the various sources
        pass

    def __call__(self):
        pass

    def get_state_dict(self) -> dict[str, Any]:
        """
        Return state dict for restart purposes
        """
        pass


# ------------------------------------------------------------------------------
# Tokens Generator
# ------------------------------------------------------------------------------


class TokenGenerator:
    """
    Generate toekn iteratively in order to feed an LLM
    """

    def __init__(self, config: DataConfig):
        pass


# %%


def read_text(file_path: str, block_size: int, offset: int) -> Generator[str, None, None]:
    with open(file_path) as file:
        file.seek(offset)
        lines = file.readlines()
        for i in range(offset, len(lines), block_size):
            yield lines[i].strip()


def tokenize(text: str) -> list[str]:
    return [ord(char) for char in text]


def pack_tokens(tokens: list[int], seq_len: int) -> Generator[list[int], None, None]:
    for i in range(0, len(tokens), seq_len):
        yield tokens[i : i + seq_len]


def batch_and_shuffle(sequences: list[list[int]], batch_size: int) -> Generator[list[list[int]], None, None]:
    batch = []
    for seq in sequences:
        batch.append(seq)
        if len(batch) == batch_size:
            # random.shuffle(batch)
            yield batch
            batch = []
    if batch:
        yield batch


def text_data_loader(
    file_path: str, block_size: int, offset: int, seq_len: int, batch_size: int
) -> Generator[list[list[int]], None, None]:
    # Step 1: Read and iterate over text file
    lines = read_text(file_path, block_size, offset)

    # Step 2: Tokenize text
    tokenized_lines = (tokenize(line) for line in lines)

    # Step 3: Pack tokens into sequences
    packed_sequences = (pack for tokens in tokenized_lines for pack in pack_tokens(tokens, seq_len))

    # Step 4: Batch and shuffle sequences
    batches = batch_and_shuffle(packed_sequences, batch_size)

    return batches


# %%
# Example usage
file_path = "/home/vivc/Code/memory/tinyshakespeare.txt"  # Path to your text file
block_size = 2
offset = 1
seq_len = 5
batch_size = 3

with open(file_path) as file:
    for _, line in enumerate(file, start=4):
        print(line)
        break

# for batch in text_data_loader(file_path, block_size, offset, seq_len, batch_size):
#     print(batch)

# %%
