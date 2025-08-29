# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Asynchronous Dataloader wrapped around a Token Generator.

### Notes
Ideally this should use PyTorch new data loading library, `torchdata`
https://pytorch.org/data/beta/what_is_torchdata_nodes.html#why-torchdata-nodes

@ 2025, Ambroise Odonnat
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from logging import getLogger
from multiprocessing import Process, Queue
from queue import Empty, Full
from types import TracebackType
from typing import Any

from torch.distributed.checkpoint.stateful import Stateful

logger = getLogger("nanollama")


@dataclass
class AsyncDataConfig:
    asynchronous: bool = True
    buffer_size: int = 4


class DataLoader(ABC, Stateful):
    """
    Asynchronous Dataloader

    Generates batches of data asynchronously, before consuming them.

    ### Parameters
    - config: configuration of the data loader.

    Usage:
    ```python
    with DataLoader(*args) as data_loader:
        for batch, _ in data_loader:
            pass
    ```
    """

    def __init__(self, config: AsyncDataConfig):
        # data loader configuration
        self.asynchronous = config.asynchronous

        # asynchronous data loader: a worker writes batches in a buffer, that a reader consumes
        if self.asynchronous:
            self.async_state_dict = self.writer_state_dict()
            self.process = Process(target=self.write_batch)
            self.buffer = Queue(maxsize=config.buffer_size)

        # initialize the batch generator
        self.batch_generator = self.batch_iterator()

    def __enter__(self) -> "DataLoader":
        """Enter the data loader context."""
        logger.info("Entering dataloader.")
        if self.asynchronous:
            self.process.start()
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """Exit the data loador context."""
        logger.info("Exiting dataloader.")
        if self.asynchronous:
            self.process.kill()
            self.buffer.close()

    @abstractmethod
    def batch_iterator(self) -> Generator[Any, None, None]:
        """Generate batch of data to be bufferized."""
        ...

    @abstractmethod
    def writer_state_dict(self) -> None:
        """State of the bufferized data writter."""
        ...

    @abstractmethod
    def load_writer_state_dict(self, state_dict: dict) -> None:
        """Reload the state of the bufferized data writter."""
        ...

    def write_batch(self) -> None:
        """Asynchronous batch generation, writting batches to the buffer."""
        # loop on batch creation
        while True:
            try:
                batch = next(self.batch_generator)
                state_dict = self.writer_state_dict()
            # handle end of data asynchrounously
            except StopIteration:
                batch = state_dict = None

            # put it in the buffer
            while True:
                try:
                    self.buffer.put((batch, state_dict), timeout=0.1)
                    break
                # if the buffer is full, wait until there is space
                except Full:
                    logger.debug("Buffer is full. Waiting for data comsumption.")
            logger.debug("New batch put in the buffer.")

    def read_batch(self) -> Any:
        """Asynchronous batch acquisition, reading batches from the buffer."""
        # read batch from the buffer
        while True:
            try:
                return self.buffer.get(timeout=0.1)
            # if the buffer is full, wait until it is filled
            except Empty:
                logger.debug("Buffer is empty. Waiting for data.")

    def __iter__(self) -> "DataLoader":
        """Iterator over batches."""
        return self

    def __next__(self) -> Any:
        """Get next batch of data."""
        if self.asynchronous:
            batch, self.async_state_dict = self.read_batch()
            # handle end of data asynchrounously
            if batch is None:
                raise StopIteration
        else:
            batch = next(self.batch_generator)
        return batch

    def state_dict(self) -> dict:
        """State of the data reader."""
        if self.asynchronous:
            return self.async_state_dict
        else:
            return self.writer_state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Reload the state of the data reader."""
        self.load_writer_state_dict(state_dict)
