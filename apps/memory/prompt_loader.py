# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Evaluation data loader

#### Notes
PyTorch introduces a new logic for dataloader with `torchdata`, which could improve the code.

@ 2025, Meta
"""

from collections.abc import Generator
from dataclasses import dataclass, field
from logging import getLogger
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty, Full
from types import TracebackType
from typing import Any

from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh

from src.nanollama.data.text import JSONLIterator, SourceConfig

logger = getLogger("nanollama")


@dataclass
class DataConfig:
    """
    Data configuration for evaluation

    ### Attributes
    - source: path for the evaluation
    - batch_size: batch size
    - asynchronous: whether to use asynchronous data loading
    - buffer_size: number of batches to bufferize asynchronously for data loading
    """

    sources: list[SourceConfig] = field(default_factory=list)
    batch_size: int = 0
    asynchronous: bool = True
    buffer_size: int = 4

    def post_init(self) -> None:
        assert self.sources, "source should be specified."
        assert self.batch_size, "batch_size should be specified."


# ------------------------------------------------------------------------------
# Prompt Generator
# ------------------------------------------------------------------------------


class PromptLoader(Stateful):
    """
    Generate prompt iteratively in order to evaluate an LLM.

    ### Parameters
    - config: data configuration
    - dp_mesh: device mesh

    ### Example
    ```python
    with PromptGenerator(config) as generator:
        for batch in generator:
            print(batch)
    ```
    """

    def __init__(self, config: DataConfig, dp_mesh: DeviceMesh = None):
        super().__init__()
        self.batch_size = config.batch_size

        # initialize JSONLIterator
        if dp_mesh is None:
            rank, world_size = 0, 1
        else:
            rank, world_size = dp_mesh.get_local_rank(), dp_mesh.size()

        self.jsonl_iterators: list[JSONLIterator] = []
        for source in config.sources:
            for path in Path(source.path).glob("*.jsonl"):
                self.jsonl_iterators.append(JSONLIterator(str(path), rank, world_size, loop=False))
        self.file_idx = 0

        # asynchronous data loading: a worker writes batches in a buffer, that a reader consumes
        self.asynchronous = config.asynchronous
        if self.asynchronous:
            self.process = Process(target=self.async_batch_creator)
            self.buffer = Queue(maxsize=config.buffer_size)

        # initialize the batch generator
        self.generator = self.batch_iterator()
        self.state_dict = self.sync_state_dict()

    def __enter__(self) -> "PromptLoader":
        logger.info("Entering dataloader.")
        for iterator in self.jsonl_iterators:
            iterator.__enter__()
        if self.asynchronous:
            self.process.start()
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        if self.asynchronous:
            self.process.kill()
            self.buffer.close()
        for iterator in self.jsonl_iterators:
            iterator.__exit__(exc, value, tb)

    def batch_iterator(self) -> Generator[tuple[list[str], list[str]], None, None]:
        nb_files = len(self.jsonl_iterators)
        while True:
            prompts = []
            answers = []
            while len(prompts) < self.batch_size:
                # check the current iterator
                if self.file_idx >= nb_files:
                    break
                jsonl_iterator = self.jsonl_iterators[self.file_idx]
                try:
                    # Receive sentences and put them in the current batch
                    json_data = next(jsonl_iterator)
                    dialog = json_data["dialog"]
                    message = dialog[0]
                    assert message["source"] == "user"
                    prompts.append(message["content"])
                    answers.append(json_data["answer"])
                except StopIteration:
                    # Move to the next iterator
                    self.file_idx += 1
            # if a batch was built, yield it
            if prompts:
                yield prompts, answers
            # otherwise, stop the generator
            else:
                return

    def sync_state_dict(self) -> dict[str, Any]:
        return {"iterators": [iterator.state_dict() for iterator in self.jsonl_iterators], "file_idx": self.file_idx}

    # --------------------------------------------------------------------------
    # Asynchronous Data Loading
    # --------------------------------------------------------------------------

    def async_batch_creator(self) -> None:
        """Asynchronous batch generation, writting batches to the buffer."""
        # loop on batch creation
        while True:
            try:
                batch = next(self.generator)
                gen_state_dict = self.sync_state_dict()

            # handle end of data asynchrounously
            except StopIteration:
                batch = gen_state_dict = None

            # put it in the buffer
            while True:
                try:
                    self.buffer.put((batch, gen_state_dict), timeout=0.1)
                    break
                # if the buffer is full, wait until there is space
                except Full:
                    ...
            logger.debug("New batch put in the buffer.")

    def async_get_batch(self) -> tuple[list[str], list[str]]:
        """Asynchronous batch acquisition, reading batches from the buffer."""
        # read batch from the buffer
        while True:
            try:
                return self.buffer.get(timeout=0.1)
            # if the buffer is full, wait until it is filled
            except Empty:
                logger.debug("Buffer is empty. Waiting for data.")

    def __iter__(self) -> "PromptLoader":
        """Return an iterator over batches"""
        return self

    def __next__(self) -> tuple[list[str], list[str]]:
        """Get the next batch of sentences."""
        if self.asynchronous:
            batch, self.gen_state_dict = self.async_get_batch()
            # handle end of data asynchrounously
            if batch is None:
                raise StopIteration
        else:
            batch = next(self.generator)
            self.gen_state_dict = self.sync_state_dict()
        return batch

    def state_dict(self) -> dict:
        return self.gen_state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        # stop the running process
        if self.asynchronous:
            self.process.kill()
            # empty the buffer
            while not self.buffer.empty():
                self.buffer.get()

        # reset the generator
        for iterator, value in zip(self.jsonl_iterators, state_dict["iterators"]):
            iterator.load_state_dict(value)
        self.file_idx = state_dict["file_idx"]
        self.gen_state_dict = self.sync_state_dict()

        # restart the process
        if self.asynchronous:
            self.process = Process(target=self.async_batch_creator)
            self.process.start()
