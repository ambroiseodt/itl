"""
Text loader

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import json
from collections.abc import Generator
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import numpy as np
from torch.distributed.checkpoint.stateful import Stateful

from nanollama.data.tokenizer import Tokenizer, TokenizerConfig

# from ..distributed import get_rank, get_world_size
# from .tokenizer import Tokenizer, TokenizerConfig
# from .utils import generate_seeds

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
    batch_size: batch size
    seq_len: sequence length
    padding: whether to concatenate various sources into a sequence of tokens
    asynchronous: whether to use asynchronous data loading
    buffer_size: number of batches to bufferize asynchronously for data loading
    """

    sources: list[SourceConfig]
    tokenizer: TokenizerConfig

    batch_size: int = 0
    seq_len: int = 0
    padding: bool = False
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

    def iterator(self) -> Generator[dict[str, str], None, None]:
        while True:
            # read the file and yield lines according to the rank
            while line := self.file.readline():
                self.line_num += 1
                if (self.line_num - 1) % self.world_size == self.rank:
                    self.position = self.file.tell()
                    yield json.loads(line)

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
# Tokens Generator
# ------------------------------------------------------------------------------


class TokenGenerator(Stateful):
    """
    Generate token iteratively in order to feed an LLM

    ### Parameters
    config: data configuration
    iterator: JSONLIterator
    tokenizer: tokenizer

    ### Attributes
    tokens: list of tokens that have not populated a batch yet
    bsz: batch size
    seq_len: sequence length
    padding: whether to pad sequences

    ### Example
    ```python
    with TokenGenerator(config, iterator, tokenizer) as generator:
        for batch in generator:
            print(batch)
    ```
    """

    def __init__(self, config: DataConfig, iterator: JSONLIterator, tokenizer: Tokenizer):
        self.jsonl_iterator = iterator
        self.tokenizer = tokenizer

        self.tokens = []
        self.bsz = config.batch_size
        self.seq_len = config.seq_len
        self.tokens_per_batch = self.bsz * self.seq_len
        self.padding = config.padding

        self.generator = self.iterator()

    def __enter__(self):
        self.jsonl_iterator.__enter__()
        return self

    def __next__(self):
        return next(self.generator)

    def iterator(self) -> Generator[np.ndarray[int], None, None]:
        while True:
            while len(self.tokens) < self.tokens_per_batch:
                # receive sentences
                json_data = next(self.jsonl_iterator)
                seq = json_data["text"]

                # tokenize sequences that are received
                self.tokens.extend(self.tokenizer.encode(seq))

                # pad sequences
                if self.padding:
                    self.tokens.extend([self.tokenizer.pad_id] * (-len(self.tokens) % self.seq_len))

            batch = np.array(self.tokens[: self.tokens_per_batch]).reshape(self.bsz, self.seq_len)
            self.tokens = self.tokens[self.tokens_per_batch :]

            yield batch

    def state_dict(self) -> dict[str, Any]:
        return {"iterator": self.jsonl_iterator.state_dict(), "tokens": self.tokens}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.jsonl_iterator.load_state_dict(state_dict["iterator"])
        self.tokens = state_dict["tokens"]

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        self.jsonl_iterator.__exit__(exc, value, tb)


# ------------------------------------------------------------------------------
# Text Generator from Multiple Sources (TODO)
# ------------------------------------------------------------------------------

# In usual implementation, data is pulled from multiple sources.
# We do not need to care for it in our current project.

# Note that in the classical datasetup, each data are visited once,
# which means that we do not need to care about randomness of samples within an epoch.
# We may want to modify the code above to allow for random batch ordering.
# One way to do it is to pull all the data in memory and read from them (inline with classical ML).
# Another was to do it is to create many biography files, and pick them randomly (inline with LLM training).

# ------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile
    import unittest
    from unittest.mock import MagicMock

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
            self.assertEqual(lines, [{"key": "value1"}, {"key": "value3"}, {"key": "value5"}])

            # Test with world_size=2, rank=1
            with JSONLIterator(path=self.temp_file.name, rank=1, world_size=2) as iterator:
                lines = [next(iterator) for _ in range(2)]
            self.assertEqual(lines, [{"key": "value2"}, {"key": "value4"}])

        def test_state_dict(self) -> None:
            # Test state saving and loading
            with JSONLIterator(path=self.temp_file.name, rank=0, world_size=2) as iterator:
                [next(iterator) for _ in range(6)]  # Read first line
                state = iterator.state_dict()

            # Create a new iterator and load the state
            with JSONLIterator(path=self.temp_file.name, rank=0, world_size=2) as new_iterator:
                new_iterator.load_state_dict(state)
                line = next(new_iterator)
            self.assertEqual(line, {"key": "value3"})

        def test_loop(self) -> None:
            # Test looping functionality
            with JSONLIterator(path=self.temp_file.name, rank=0, world_size=2) as iterator:
                lines = [next(iterator) for _ in range(6)]
            self.assertEqual(
                lines,
                [
                    {"key": "value1"},
                    {"key": "value3"},
                    {"key": "value5"},
                    {"key": "value2"},
                    {"key": "value4"},
                    {"key": "value1"},
                ],
            )

    class TestTokenGenerator(unittest.TestCase):
        def setUp(self) -> None:
            # Mock the JSONLIterator and Tokenizer
            self.mock_tokenizer = MagicMock()

            self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w+")
            self.temp_file.write('{"text": "sentence one"}\n')
            self.temp_file.write('{"text": "sentence two"}\n')
            self.temp_file.write('{"text": "sentence three"}\n')
            self.temp_file.write('{"text": "sentence four"}\n')
            self.temp_file.write('{"text": "sentence five"}\n')
            self.temp_file.flush()
            self.temp_file.seek(0)

            self.iterator = JSONLIterator(path=self.temp_file.name, rank=0, world_size=1)
            self.iterator.__enter__()

            # Mock data
            self.mock_tokenizer.encode.side_effect = lambda x: [ord(c) for c in x]
            self.mock_tokenizer.pad_id = 0

            # Configuration
            self.config = DataConfig(
                sources=None,
                tokenizer=None,
                batch_size=2,
                seq_len=5,
                padding=True,
            )

            # Initialize TokenGenerator
            self.token_generator = TokenGenerator(self.config, self.iterator, self.mock_tokenizer)

        def tearDown(self) -> None:
            # Close and remove the temporary file
            self.iterator.__exit__(None, None, None)
            self.temp_file.close()
            os.unlink(self.temp_file.name)

        def test_token_generation(self) -> None:
            # Test if the generator produces the expected output
            batch = next(self.token_generator)
            expected_shape = (self.config.batch_size, self.config.seq_len)
            self.assertEqual(batch.shape, expected_shape)

        def test_padding(self) -> None:
            # Test if padding is applied correctly
            next(self.token_generator)
            batch = next(self.token_generator)
            self.assertTrue(batch[0, -1] == self.mock_tokenizer.pad_id)

        def test_state_dict(self) -> None:
            # Restart from previous state_dict
            initial_state = self.token_generator.state_dict()
            batch = next(self.token_generator)
            next(self.token_generator)

            # Create a new TokenGenerator and load the saved state
            new_token_generator = TokenGenerator(self.config, self.iterator, self.mock_tokenizer)
            new_token_generator.load_state_dict(initial_state)
            # Generate a batch from the new generator
            restarted_batch = next(new_token_generator)
            assert np.array_equal(batch, restarted_batch)

    unittest.main()
