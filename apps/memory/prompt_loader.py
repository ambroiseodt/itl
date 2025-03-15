# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Evaluation data loader

#### Notes
PyTorch introduces a new logic for dataloader with `torchdata`, which could improve the code.

@ 2025, Meta
"""

from collections.abc import Generator
from logging import getLogger
from pathlib import Path
from types import TracebackType
from typing import Any

from torch.distributed.device_mesh import DeviceMesh

from src.nanollama.data import DataLoader
from src.nanollama.data.text import DataConfig, JSONLIterator
from src.nanollama.data.tokenizer import Actor, build_tokenizer

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# Prompt Generator
# ------------------------------------------------------------------------------


class PromptLoader(DataLoader):
    """
    Generate prompt iteratively in order to evaluate an LLM.

    ### Parameters
    - config: data configuration
    - dp_mesh: device mesh

    ### Example
    ```python
    with PromptLoader(config) as generator:
        for batch in generator:
            print(batch)
    ```
    """

    def __init__(self, config: DataConfig, dp_mesh: DeviceMesh = None):
        super().__init__(config)
        self.batch_size = config.batch_size

        self.tokenizer = build_tokenizer(config.tokenizer)

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

    def __enter__(self) -> "PromptLoader":
        logger.info("Entering dataloader.")
        for iterator in self.jsonl_iterators:
            iterator.__enter__()
        super().__enter__()
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        super().__exit__(exc, value, tb)
        for iterator in self.jsonl_iterators:
            iterator.__exit__(exc, value, tb)

    def batch_iterator(self) -> Generator[tuple[list[list[int]], list[str]], None, None]:
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
                    assert dialog[0]["source"] == "user"
                    tokens, _ = self.tokenizer.encode(dialog[:1])
                    tokens.append(self.tokenizer.bots[Actor.assistant])
                    prompts.append(tokens)
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

    def writer_state_dict(self) -> dict[str, Any]:
        return {"iterators": [iterator.state_dict() for iterator in self.jsonl_iterators], "file_idx": self.file_idx}

    def load_writer_state_dict(self, state_dict: dict) -> None:
        for iterator, value in zip(self.jsonl_iterators, state_dict["iterators"]):
            iterator.load_state_dict(value)
        self.file_idx = state_dict["file_idx"]
