# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Evaluation data loader

#### Notes
PyTorch introduces a new logic for dataloader with `torchdata`, which could improve the code.

@ 2025, Meta
"""

from collections.abc import Generator
from dataclasses import dataclass
from logging import getLogger
from multiprocessing import Process, Queue
from queue import Empty, Full
from types import TracebackType
from typing import Any

from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh

from nanollama.data.text import JSONLIterator

# from src.nanollama.data.text import JSONLIterator

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

    source: str
    batch_size: int
    asynchronous: bool = True
    buffer_size: int = 4


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
        self.jsonl_iterator = JSONLIterator(config.source, rank, world_size, loop=False)

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
        self.jsonl_iterator.__enter__()
        if self.asynchronous:
            self.process.start()
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        self.jsonl_iterator.__exit__(exc, value, tb)

    def batch_iterator(self) -> Generator[tuple[list[str], list[str]], None, None]:
        while True:
            try:
                prompts = []
                answers = []
                for _ in range(self.batch_size):
                    # receive sentences
                    json_data = next(self.jsonl_iterator)
                    dialog = json_data["dialog"]

                    message = dialog[0]
                    assert message["source"] == "user"
                    prompts.append(message["content"])
                    answers.append(json_data["answer"])
                yield prompts, answers
            except StopIteration:
                if len(prompts):
                    yield prompts, answers
                else:
                    break

    def sync_state_dict(self) -> dict[str, Any]:
        return {"iterator": self.jsonl_iterator.state_dict()}

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
                    logger.debug("Buffer is full. Waiting for data comsumption.")
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
        self.jsonl_iterator.load_state_dict(state_dict["iterator"])
        self.gen_state_dict = self.sync_state_dict()

        # restart the process
        if self.asynchronous:
            self.process = Process(target=self.async_batch_creator)
            self.process.start()


if __name__ == "__main__":
    config = DataConfig(
        source="/private/home/vivc/code/memory/apps/memory/dataset/qatool.jsonl",
        batch_size=10,
        asynchronous=False,
        buffer_size=4,
    )

    data_loader = PromptLoader(config)
    with data_loader:
        for batch in data_loader:
            print(batch)
            break
