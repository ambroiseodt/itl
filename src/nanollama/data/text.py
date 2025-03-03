# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Text loader

#### Notes
PyTorch introduces a new logic for dataloader with `torchdata`.
The various classes defined in this modules are made to be cascaded.
In `torchdata`, this logic will be replaced by operators applied to a single TorchData object to cascade operations.

@ 2025, Meta
"""

import json
import os
from collections.abc import Generator
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import numpy as np
from numpy.random import default_rng
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh

from .loader import TokenLoader
from .tokenizer import DialogTokenizer, TokenizerConfig, build_tokenizer
from .utils import generate_seeds

# ------------------------------------------------------------------------------
# Configuration classes
# ------------------------------------------------------------------------------


@dataclass
class SourceConfig:
    """
    Text source configuration

    ### Attributes
    - path: path to text files
    - weight: weighting of the source in the datamix
    """

    path: str
    weight: float

    def __post_init__(self):
        self.path = os.path.expandvars(self.path)


@dataclass
class DataConfig:
    """
    Data configuration for text data loader

    ### Attributes
    - source: corpus of text specification as a list of weighted sources
    - tokenizer: tokenizer configuration
    - batch_size: batch size
    - seq_len: sequence length
    - padding: whether to concatenate various sources into a sequence of tokens
    - asynchronous: whether to use asynchronous data loading
    - buffer_size: number of batches to bufferize asynchronously for data loading
    """

    sources: list[SourceConfig]
    tokenizer: TokenizerConfig

    batch_size: int
    seq_len: int
    padding: bool = False
    seed: int = 0
    asynchronous: bool = True
    buffer_size: int = 4


# ------------------------------------------------------------------------------
# Text Generator from JSONL
# ------------------------------------------------------------------------------


class JSONLIterator(Stateful):
    """
    Iterates over a JSON lines file, yielding a line every `world_size` lines

    ### Parameters
    - path: filepath
    - rank: rank of the worker
    - world_size: number of workers
    - loop: whether to loop over the file

    ### Attributes
    - position: current position in the file
    - line_num: current line number
    - generator: generator that yields lines
    - loop: current loop number (useful to check how many times data were seen)

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
        self.loop = int(loop)

        self.file = None
        self.position = 0
        self.line_num = 0

        self.generator = self.line_iterator()

    def __enter__(self) -> "JSONLIterator":
        self.file = open(self.path)
        self.file.seek(self.position)
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        self.file.close()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    def line_iterator(self) -> Generator[dict[str, str], None, None]:
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
            self.loop += 1
            self.file.seek(0)

    def state_dict(self) -> dict[str, Any]:
        return {"line_num": self.line_num, "position": self.position}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.line_num = state_dict["line_num"]
        self.position = state_dict["position"]
        self.file.seek(self.position)


# ------------------------------------------------------------------------------
# Tokens Generator
# ------------------------------------------------------------------------------


class SingleSourceTokenGenerator(TokenLoader):
    """
    Generate token iteratively in order to feed an LLM based on a single sequence iterator.

    Pulls sequences from the sequence iterator, and transform them into tokens.
    Yield batches of tokens with shape `bsz x seq_len`.

    ### Parameters
    - config: data configuration
    - iterator: sequence (string) iterator
    - tokenizer: tokenizer

    ### Attributes
    - tokens: list of tokens that have not populated a batch yet
    - mask: list of specifying tokens produced by the LLM assistant
    - bsz: batch size
    - seq_len: sequence length
    - padding: whether to pad sequences

    ### Methods
    - set_batch_size: modify the batch size of the generator

    ### Example
    ```python
    with TokenGenerator(config, iterator, tokenizer) as generator:
        for batch in generator:
            print(batch)
    ```
    """

    def __init__(self, config: DataConfig, iterator: JSONLIterator, tokenizer: DialogTokenizer):
        super().__init__()
        self.jsonl_iterator = iterator
        self.tokenizer = tokenizer

        self.tokens = []
        self.mask = []
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.tokens_per_batch = self.batch_size * self.seq_len
        self.padding = config.padding

    def __enter__(self) -> "SingleSourceTokenGenerator":
        self.jsonl_iterator.__enter__()
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        self.jsonl_iterator.__exit__(exc, value, tb)

    def set_batch_size(self, batch_size: int) -> None:
        """
        Modify the batch size of the generator

        ### Parameters
        - batch_size: new batch size
        """
        self.batch_size = batch_size
        self.tokens_per_batch = self.batch_size * self.seq_len

    def batch_iterator(self) -> Generator[tuple[np.ndarray[int], np.ndarray[bool]], None, None]:
        while True:
            while len(self.tokens) < self.tokens_per_batch:
                # receive sentences
                json_data = next(self.jsonl_iterator)
                dialog = json_data["dialog"]

                # tokenize sequences that are received
                tokens, mask = self.tokenizer.encode(dialog)
                self.tokens.extend(tokens)
                self.mask.extend(mask)

                # pad sequences
                if self.padding:
                    ext_len = -len(self.tokens) % self.seq_len
                    self.tokens.extend([0] * ext_len)
                    self.mask.extend([False] * ext_len)

            batch = np.array(self.tokens[: self.tokens_per_batch]).reshape(self.batch_size, self.seq_len)
            self.tokens = self.tokens[self.tokens_per_batch :]

            mask = np.array(self.mask[: self.tokens_per_batch]).reshape(self.batch_size, self.seq_len)
            self.mask = self.mask[self.tokens_per_batch :]

            yield batch, mask

    def state_dict(self) -> dict[str, Any]:
        return {"iterator": self.jsonl_iterator.state_dict(), "tokens": self.tokens, "mask": self.mask}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.jsonl_iterator.load_state_dict(state_dict["iterator"])
        self.tokens = state_dict["tokens"]
        self.mask = state_dict["mask"]


# ------------------------------------------------------------------------------
# Text Generator from Multiple Sources
# ------------------------------------------------------------------------------


class MultipleSourcesTokenGenerator(TokenLoader):
    """
    Generate token from multiple single source iterators according to some weight.

    ### Parameters
    - config: data configuration
    - dp_mesh: device mesh

    ### Attributes
    - generators: list of single source token generators
    - weights: weights of the different sources in the datamix
    - rng: random number generator

    ### Example
    ```python
    with MultipleSourceTokenGenerator(config) as generator:
        for batch in generator:
            print(batch)
    ```
    """

    def __init__(self, config: DataConfig, dp_mesh: DeviceMesh = None):
        super().__init__()

        self.batch_size = config.batch_size
        self.seq_len = config.seq_len

        # initialize the single source iterators
        tokenizer = build_tokenizer(config.tokenizer)
        if dp_mesh is None:
            rank, world_size = 0, 1
        else:
            rank, world_size = dp_mesh.get_local_rank(), dp_mesh.size()
        iterators = [JSONLIterator(source.path, rank, world_size) for source in config.sources]
        self.generators = [SingleSourceTokenGenerator(config, iterator, tokenizer) for iterator in iterators]

        # initialize the source weights
        self.weights = np.array([source.weight for source in config.sources], dtype=float)
        self.weights /= np.sum(self.weights)

        # initialize the random number generator
        _, seeds = generate_seeds(nb_shared=0, nb_individual=1, root_seed=config.seed, rank=rank)
        self.rng = default_rng(seeds[0])

    def __enter__(self) -> "MultipleSourcesTokenGenerator":
        for generator in self.generators:
            generator.__enter__()
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        for generator in self.generators:
            generator.__exit__(exc, value, tb)

    def batch_iterator(self) -> Generator[tuple[np.ndarray[int], np.ndarray[bool]], None, None]:
        while True:
            # generate a random mix of sources
            source_mix = self.rng.multinomial(self.batch_size, self.weights)

            # create a batch accordingly
            batch = np.zeros((self.batch_size, self.seq_len), dtype=int)
            mask = np.zeros((self.batch_size, self.seq_len), dtype=int)
            ind = 0
            for gen, bsz in zip(self.generators, source_mix):
                gen.set_batch_size(bsz)
                local_batch, local_mask = next(gen)
                batch[ind : ind + bsz] = local_batch
                mask[ind : ind + bsz] = local_mask
                ind = ind + bsz
            yield np.vstack((batch, mask), dtype=int)

    def state_dict(self) -> dict[str, Any]:
        return {"generators": [gen.state_dict() for gen in self.generators], "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for gen, state in zip(self.generators, state_dict["generators"]):
            gen.load_state_dict(state)
        self.rng.bit_generator.state = state_dict["rng_state"]
