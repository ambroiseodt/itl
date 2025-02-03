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
from types import TracebackType
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

    ### Attributes
    path: path to text files
    weight: weighting of the source in the datamix
    """

    path: str
    weight: float


@dataclass
class DataConfig:
    """
    Data configuration for text data loader

    ### Attributes
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
# Text Generator from JSONL
# ------------------------------------------------------------------------------


class JSONLIterator(Stateful):
    """
    Iterates over a JSON lines file, yielding a line every `world_size` lines

    ### Parameters
    path: filepath
    rank: rank of the worker
    world_size: number of workers
    loop: whether to loop over the file

    ### Attributes
    position: current position in the file
    line_num: current line number
    generator: generator that yields lines

    ### Example
    ```python
    with JSONLIterator("path/to/file.jsonl", 0, 2) as iterator:
        for line in iterator:
            print(line)
    ```
    If world_size = 2, rank = 0, iterator will yield lines 0 2 4 6 ...\\
    If world_size = 3, rank = 1, iterator will yield lines 1 4 7 10 ...
    """

    def __init__(
        self,
        path: str,
        rank: int,
        world_size: int,
        loop: bool = True,
    ):
        self.path = path
        self.rank = rank
        self.world_size = world_size
        self.loop = loop

        self.file = None
        self.position = 0
        self.line_num = 0

        self.generator = self.iterator()

    def __enter__(self):
        self.file = open(self.path)
        self.file.seek(self.position)
        return self

    def __next__(self):
        return next(self.generator)

    def iterator(self) -> Generator[str, None, None]:
        while True:
            # read the file and yield lines according to the rank
            while line := self.file.readline():
                self.line_num += 1
                if (self.line_num - 1) % self.world_size == self.rank:
                    self.position = self.file.tell()
                    yield line

            # potential rewind when the end of the file is reached
            if not self.loop:
                break
            self.file.seek(0)

    def state_dict(self) -> dict[str, Any]:
        return {"line_num": self.line_num, "position": self.position}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.line_num = state_dict["line_num"]
        self.position = state_dict["position"]
        self.file.seek(self.position)

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        self.file.close()


# ------------------------------------------------------------------------------
# Text Generator from Multiple Sources (TODO)
# ------------------------------------------------------------------------------

"""
No need for something too fancy at the moment

We should actually start by tokenizing the JSONL context
Then we play around with tokens (packing them in various ways).
"""


@dataclass
class DataLoaderState(Stateful):
    """
    To remove and integrate within the textgenerator class.
    """

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


class TextGenerator(Stateful):
    """
    Generate sequences iteratively in order to feed a tokenizer

    Choose sentences by iterating over multiple JSONL sources

    TODO
    """

    def __init__(self, config: DataConfig):
        # list all sources

        # weights for the various sources
        pass

    def __enter__(self):
        # open all files
        pass

    def __next__(self):
        # choose a file
        # open this file
        # seek the correct position
        # read a line
        pass

    def state_dict(self) -> dict[str, Any]:
        pass

    def load_state_dict(self) -> dict[str, Any]:
        pass

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        # close all files
        pass


# ------------------------------------------------------------------------------
# Tokens Generator (TODO)
# ------------------------------------------------------------------------------


class TokenGenerator:
    """
    Generate toekn iteratively in order to feed an LLM
    """

    def __init__(self, config: DataConfig):
        pass

    def __call__(self):
        # tokenize sequences that are received
        # make chunks of size (bsz, seq_len)
        pass


# ------------------------------------------------------------------------------
# Older stuff
# ------------------------------------------------------------------------------


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


# # Example usage
# file_path = "/home/vivc/Code/memory/tmp_tinyshakespeare.txt"  # Path to your text file
# file_path = "/checkpoint/vivc/datasets/tinyshakespeare.txt"  # Path to your text file
# block_size = 2
# offset = 1
# seq_len = 5
# batch_size = 3

# with open(file_path) as file:
#     for _, line in enumerate(file, start=4):
#         print(line)
#         break

# for batch in text_data_loader(file_path, block_size, offset, seq_len, batch_size):
#     print(batch)


if __name__ == "__main__":
    import os
    import tempfile
    import unittest

    class TestJSONLIterator(unittest.TestCase):
        def setUp(self) -> None:
            # Create a temporary JSONL file for testing
            self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w+")
            self.temp_file.write('{"key": "value1"}\n')
            self.temp_file.write('{"key": "value2"}\n')
            self.temp_file.write('{"key": "value3"}\n')
            self.temp_file.write('{"key": "value4"}\n')
            self.temp_file.write('{"key": "value5"}\n')
            self.temp_file.flush()
            self.temp_file.seek(0)

        def tearDown(self) -> None:
            # Close and remove the temporary file
            self.temp_file.close()
            os.unlink(self.temp_file.name)

        def test_iteration(self) -> None:
            # Test with world_size=2, rank=0
            with JSONLIterator(path=self.temp_file.name, rank=0, world_size=2) as iterator:
                lines = [next(iterator) for _ in range(3)]
            self.assertEqual(lines, ['{"key": "value1"}\n', '{"key": "value3"}\n', '{"key": "value5"}\n'])

            # Test with world_size=2, rank=1
            with JSONLIterator(path=self.temp_file.name, rank=1, world_size=2) as iterator:
                lines = [next(iterator) for _ in range(2)]
            self.assertEqual(lines, ['{"key": "value2"}\n', '{"key": "value4"}\n'])

        def test_state_dict(self) -> None:
            # Test state saving and loading
            with JSONLIterator(path=self.temp_file.name, rank=0, world_size=2) as iterator:
                [next(iterator) for _ in range(6)]  # Read first line
                state = iterator.state_dict()

            # Create a new iterator and load the state
            with JSONLIterator(path=self.temp_file.name, rank=0, world_size=2) as new_iterator:
                new_iterator.load_state_dict(state)
                line = next(new_iterator)
            self.assertEqual(line, '{"key": "value3"}\n')

        def test_loop(self) -> None:
            # Test looping functionality
            with JSONLIterator(path=self.temp_file.name, rank=0, world_size=2) as iterator:
                lines = [next(iterator) for _ in range(6)]
            self.assertEqual(
                lines,
                [
                    '{"key": "value1"}\n',
                    '{"key": "value3"}\n',
                    '{"key": "value5"}\n',
                    '{"key": "value2"}\n',
                    '{"key": "value4"}\n',
                    '{"key": "value1"}\n',
                ],
            )

    unittest.main()

    # tmp = TestJSONLIterator()
    # tmp.setUp()
    # tmp.test_iteration()
    # tmp.test_state_dict()
    # tmp.test_loop()
    # tmp.tearDown()
