"""
Unit tests for dataloading logic.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import unittest
from collections.abc import Generator
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import numpy as np
import torch
from numpy.random import default_rng

from nanollama.data.loader import DataLoader, TokenLoader
from nanollama.data.utils import generate_seeds


@dataclass
class MockDataConfig:
    batch_size: int
    seq_len: int
    seed: int = 0
    asynchronous: bool = True
    buffer_size: int = 4


class MockTokenLoader(TokenLoader):
    def __init__(self, config: MockDataConfig):
        super().__init__()
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.rng = default_rng(config.seed)

    def __enter__(self):
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        pass

    def batch_iterator(self) -> Generator[np.ndarray[int], None, None]:
        while True:
            batch = self.rng.integers(0, 100, size=(self.batch_size, self.seq_len))
            yield batch

    def state_dict(self) -> dict[str, Any]:
        return {"rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state_dict["rng_state"]


class TestDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MockDataConfig(batch_size=2, seq_len=5)
        self.mock_iterator = MockTokenLoader(self.config)
        self.data_loader = DataLoader(self.config, self.mock_iterator)
        self.data_loader.__enter__()

    def tearDown(self) -> None:
        self.data_loader.__exit__(None, None, None)

    def test_asynchronous_loading(self) -> None:
        # Test DataLoader in synchronous mode
        config = MockDataConfig(batch_size=2, seq_len=5, asynchronous=False)
        sync_data_loader = DataLoader(config, self.mock_iterator)

        for _ in range(3):
            batch = next(self.data_loader)
            sync_batch = next(sync_data_loader)
            self.assertEqual(batch.shape, (config.batch_size, config.seq_len))
            self.assertIsInstance(batch, torch.Tensor)
            assert np.array_equal(batch, sync_batch)

    def test_restart(self) -> None:
        # Test state saving and loading
        next(self.data_loader)
        state = self.data_loader.state_dict()
        batch = next(self.data_loader)
        next(self.data_loader)

        new_loader = DataLoader(self.config, self.mock_iterator)
        new_loader.__enter__()
        new_loader.load_state_dict(state)
        new_batch = next(new_loader)
        self.assertTrue(torch.equal(batch, new_batch))
        new_loader.__exit__(None, None, None)


class TestGenerateSeeds(unittest.TestCase):
    def test_seeds(self) -> None:
        nb_shared = 10
        nb_individual = 5
        root_seed = 42
        world_size = 16
        base_ss = None
        base_is = None
        for rank in range(world_size):
            s_s, i_s = generate_seeds(nb_shared, nb_individual, root_seed, rank)
            assert len(s_s) == nb_shared
            if base_ss is None:
                base_ss = s_s
            else:
                for base, seed in zip(base_ss, s_s):
                    assert base.entropy == seed.entropy
                    assert base.spawn_key == seed.spawn_key
            assert len(i_s) == nb_individual
            if base_is is None:
                base_is = i_s
            else:
                for base, seed in zip(base_is, i_s):
                    assert base.entropy == seed.entropy
                    assert base.spawn_key != seed.spawn_key
