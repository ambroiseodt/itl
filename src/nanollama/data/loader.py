"""
Dataloader

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from logging import getLogger
from multiprocessing import Process, Queue
from queue import Empty, Full
from types import TracebackType
from typing import Any

import numpy as np
import torch
from torch.distributed.checkpoint.stateful import Stateful

logger = getLogger("nanollama")


class TokenLoader(ABC, Stateful):
    def __init__(self):
        self.generator = self.batch_iterator()

    def __next__(self) -> np.ndarray:
        """Get the next batch of tokens."""
        return next(self.generator)

    @abstractmethod
    def __enter__(self) -> Generator[np.ndarray, None, None]:
        """Enter the batch generator context (opening files, ...)."""
        ...

    @abstractmethod
    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """Exit the batch generator context (closing files, ...)."""
        ...

    @abstractmethod
    def batch_iterator(self) -> Generator[np.ndarray, None, None]:
        """Generate batches of tokens."""
        ...


@dataclass
class DataConfig:
    asynchronous: bool = True
    buffer_size: int = 4


class DataLoader(Stateful):
    """
    Asynchronous Dataloader

    ### Parameters
    config: configuration of the data loader.
    generator: token loader to use to generate batches.

    Usage:
    ```python
    with DataLoader(*args) as data_loader:
        for batch, _ in data_loader:
            pass
    ```
    """

    def __init__(self, config: DataConfig, generator: TokenLoader):
        # data loader configuration
        self.asynchronous = config.asynchronous

        # asynchronous data loader: a worker writes batches in a buffer, that a reader consumes
        if self.asynchronous:
            self.process = Process(target=self.async_batch_creator)
            self.buffer = Queue(maxsize=config.buffer_size)

        # initialize the batch generator
        self.generator = generator
        self.gen_state_dict = self.generator.state_dict()

    def __enter__(self):
        """Enter the data generator context."""
        logger.info("Entering dataloader.")
        if self.asynchronous:
            self.process.start()
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """Exit the data generator context."""
        logger.info("Exiting dataloader.")
        if self.asynchronous:
            self.process.kill()
            self.buffer.close()

    def async_batch_creator(self) -> None:
        """Asynchronous batch generation, writting batches to the buffer."""
        # loop on batch creation
        while True:
            try:
                batch = next(self.generator)
                gen_state_dict = self.generator.state_dict()
                batch = torch.from_numpy(batch).long()
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
                    logger.debug("Buffer is full. Waiting for data comsumption.")
            logger.debug("New batch put in the buffer.")

    def async_get_batch(self) -> tuple[np.ndarray, tuple]:
        """Asynchronous batch acquisition, reading batches from the buffer."""
        # read batch from the buffer
        while True:
            try:
                return self.buffer.get(timeout=0.1)
            # if the buffer is full, wait until it is filled
            except Empty:
                logger.debug("Buffer is empty. Waiting for data.")

    def __iter__(self) -> "DataLoader":
        """Return an iterator over batches"""
        return self

    def __next__(self) -> tuple[torch.Tensor, Any]:
        """Get the next batch of sentences."""
        if self.asynchronous:
            batch, self.gen_state_dict = self.async_get_batch()
            # handle end of data asynchrounously
            if batch is None:
                raise StopIteration
        else:
            batch = next(self.generator)
            self.gen_state_dict = self.state_dict()
            batch = torch.from_numpy(batch).long()
        return batch

    def state_dict(self) -> dict:
        return self.gen_state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        self.gen_state_dict = state_dict
