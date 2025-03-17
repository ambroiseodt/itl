# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Unit tests for text dataloading.

@ 2025, Meta
"""

import os
import tempfile
import unittest

import numpy as np

from nanollama.data.text import (
    DataConfig,
    JSONLIterator,
    SingleSourceTokenGenerator,
    SourceConfig,
    TokenLoader,
)
from nanollama.tokenizer import build_tokenizer


class MockTokenizer:
    pad_id: int = 0

    def __init__(self):
        pass

    def encode(self, sequence: str) -> list[int]:
        data = [ord(c) for c in sequence[0]["content"]]
        mask = [True for _ in data]
        return data, mask

    def decode(self, tokens: list[int]) -> str:
        return "".join([chr(c) for c in tokens])


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

    def test_restart(self) -> None:
        # Test state saving and loading
        with JSONLIterator(path=self.temp_file.name, rank=0, world_size=2) as iterator:
            [next(iterator) for _ in range(6)]  # Read first lines
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


class TestSingleSourceTokenGenerator(unittest.TestCase):
    def setUp(self) -> None:
        # Create a temporary JSONL file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w+")
        self.temp_file.write('{"dialog": [{"source": "assistant", "content": "sentence one"}]}\n')
        self.temp_file.write('{"dialog": [{"source": "assistant", "content": "sentence two"}]}\n')
        self.temp_file.write('{"dialog": [{"source": "assistant", "content": "sentence three"}]}\n')
        self.temp_file.write('{"dialog": [{"source": "assistant", "content": "sentence four"}]}\n')
        self.temp_file.write('{"dialog": [{"source": "assistant", "content": "sentence five"}]}\n')
        self.temp_file.flush()
        self.temp_file.seek(0)

        # Mock the JSONLIterator and Tokenizer
        self.iterator = JSONLIterator(path=self.temp_file.name, rank=0, world_size=1)
        self.mock_tokenizer = MockTokenizer()

        # Configuration
        self.config = DataConfig(
            sources="fake",
            tokenizer="fake",
            batch_size=2,
            seq_len=5,
            padding=True,
            asynchronous=False,
        )

        # Initialize TokenGenerator
        self.token_generator = SingleSourceTokenGenerator(
            batch_size=self.config.batch_size,
            seq_len=self.config.seq_len,
            padding=self.config.padding,
            iterator=self.iterator,
            tokenizer=self.mock_tokenizer,
        )
        self.token_generator.__enter__()

    def tearDown(self) -> None:
        # Close and remove the temporary file
        self.token_generator.__exit__(None, None, None)
        self.temp_file.close()
        os.unlink(self.temp_file.name)

    def test_token_generation(self) -> None:
        # Test if the generator produces the expected output
        batch = next(self.token_generator)[0]
        expected_shape = (self.config.batch_size, self.config.seq_len)
        self.assertEqual(batch.shape, expected_shape)

    def test_padding(self) -> None:
        # Test if padding is applied correctly
        next(self.token_generator)
        batch = next(self.token_generator)[0]
        self.assertTrue(batch[0, -1] == 0)

    def test_restart(self) -> None:
        # Restart from previous state_dict
        initial_state = self.token_generator.state_dict()
        batch = next(self.token_generator)[0]
        next(self.token_generator)

        # Create a new TokenGenerator and load the saved state
        new_token_generator = SingleSourceTokenGenerator(
            batch_size=self.config.batch_size,
            seq_len=self.config.seq_len,
            padding=self.config.padding,
            iterator=self.iterator,
            tokenizer=self.mock_tokenizer,
        )
        new_token_generator.load_state_dict(initial_state)
        # Generate a batch from the new generator
        restarted_batch = next(new_token_generator)[0]
        assert np.array_equal(batch, restarted_batch)

    def test_bsz(self) -> None:
        # Test if changing the batch size works
        initial_state = self.token_generator.state_dict()
        n = 10
        self.token_generator.set_batch_size(1)
        for _ in range(n):
            batch = next(self.token_generator)[0]
        assert batch.shape == (1, self.config.seq_len)

        self.token_generator.load_state_dict(initial_state)
        self.token_generator.set_batch_size(n)
        new_batch = next(self.token_generator)[0]
        assert new_batch.shape == (n, self.config.seq_len)
        assert np.array_equal(batch, new_batch[-1:])


class TestTokenGenerator(unittest.TestCase):
    def setUp(self) -> None:
        # Create temporary JSONL files for testing
        self.temp_files: list[tempfile._TemporaryFileWrapper] = []
        for i in range(2):
            temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w+")
            temp_file.write('{"dialog": [{"source": "assistant", "content": "sentence one, file ' + str(i) + '"}]}\n')
            temp_file.write('{"dialog": [{"source": "assistant", "content": "sentence two, file ' + str(i) + '"}]}\n')
            temp_file.flush()
            temp_file.seek(0)
            self.temp_files.append(temp_file)
        # Configuration
        self.config = DataConfig(
            sources=[
                SourceConfig(path=self.temp_files[0].name, weight=0.5),
                SourceConfig(path=self.temp_files[1].name, weight=0.5),
            ],
            batch_size=2,
            seq_len=5,
            padding=True,
            seed=42,
            asynchronous=False,
        )
        self.tokenizer = (build_tokenizer({"implementation": "byte"}),)
        # Initialize TokenLoader
        self.token_generator = TokenLoader(self.config, self.tokenizer)
        self.token_generator.__enter__()

    def tearDown(self) -> None:
        # Close and remove the temporary files
        self.token_generator.__exit__(None, None, None)
        for temp_file in self.temp_files:
            temp_file.close()
            os.unlink(temp_file.name)

    def test_token_generation(self) -> None:
        # Test if the generator produces the expected output
        batch = next(self.token_generator)
        expected_shape = (2 * self.config.batch_size, self.config.seq_len)
        self.assertEqual(batch.shape, expected_shape)

    def test_restart(self) -> None:
        # Test state saving and loading
        initial_state = self.token_generator.state_dict()
        batch = next(self.token_generator)

        # Create a new generator and load the saved state
        new_token_generator = TokenLoader(self.config, self.tokenizer)
        new_token_generator.__enter__()
        new_token_generator.load_state_dict(initial_state)

        # Generate a batch from the new generator
        restarted_batch = next(new_token_generator)
        assert np.array_equal(batch, restarted_batch)
        new_token_generator.__exit__(None, None, None)
