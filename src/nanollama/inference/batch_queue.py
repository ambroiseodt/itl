# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Module to generate completions in parallel with a queuing mechanism when lengths differs.

@ 2025, Meta

### Notes
This implementation is minimalistic, in particular it makes synchronous calls to the SQL agent.
In practical applications, a agentic call should be asynchronous to avoid blocking the generation process.
Some page-in/page-out mechanism should be implemented to ensure maximum throughput in the meantime.
"""

from types import TracebackType

import torch

from ..agent import Actor, SQLAgent
from ..data.tokenizer import DialogTokenizer
from ..model import Transformer


class QueuedBatchedInference:
    """
    Batched inference with queues.

    ### Parameters
    - model: transformer model
    - tokenizer: tokenizer
    - db_path: path to the database containing facts

    ### Attributes
    - queue: queues of tokens to be added to the context
    """

    def __init__(self, model: Transformer, tokenizer: DialogTokenizer, db_path: str):
        # Attributes for generation and decoding
        self.model = model
        self.tokenizer = tokenizer
        self.queue: list[list[int]]
        self.agent = SQLAgent(db_path)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def __enter__(self):
        self.agent.__enter__()
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        self.agent.__exit__(exc, value, tb)

    def generate(self, prompts: list[str]) -> list[str]:
        """
        Generate completions for the given prompts.

        ### Parameters
        - prompts: list of prompts

        ### Returns
        - output: list of completions
        """
        # aliases
        bot2actor = self.tokenizer.bot2actor
        decode = self.tokenizer.tokenizer.decode
        encode = self.tokenizer.tokenizer.encode
        assistant = self.tokenizer.bots[Actor.assistant]

        # prepare generation
        x = self.build_batch(prompts)
        bsz, total_len = x.size()
        self.model.build_kv_cache(bsz)

        output = []
        buffers = [[] for _ in prompts]

        while total_len < self.model.seq_len:
            preds = self.model(x)
            x = preds[:, -1:].argmax(dim=-1)

            # inspect each lane
            for i in range(bsz):
                # replace token that were set in advanced
                if len(self.queue[i]) > 0:
                    x[i, 0] = self.queue[i].pop(0)
                    continue

                # check if the LLM is calling a tool
                token = x[i, 0].item()
                actor = bot2actor.get(token, None)

                # if so, decode the current LLM message, seen as instructions
                if actor is not None:
                    instructions = decode(buffers[i])
                    buffers[i] = []

                # ask the agent to execute itself based on the instructions
                if actor == self.agent.actor:
                    answer = self.agent.execute(instructions)
                    # encode its answer and add it to the queue
                    self.queue[i].extend(encode(answer))
                    # and call assistant turn
                    self.queue[i].append(assistant)

            total_len += 1
            output.append(x)

        output = torch.hstack(output)
        return [self.tokenizer.decode(out.tolist()) for out in output]

    def build_batch(self, prompts: list[str]) -> torch.Tensor:
        """
        Build the batch for the model.

        ### Parameters
        - prompts: list of prompts

        ### Returns
        - input_ids: input tensor
        - batch_offset: position of the first token of each prompt
        """
        data = [self.encode_prompt(prompt) for prompt in prompts]
        bsz = len(data)
        seq_len = min([len(datum) for datum in data])
        dtype, device = torch.long, self.model.device

        x = torch.zeros((bsz, seq_len), dtype=dtype, device=device)
        self.queue = [[] for _ in prompts]
        for i, datum in enumerate(data):
            x[i, :seq_len] = datum[:seq_len]
            self.queue[i] = datum[seq_len:]
        return x

    def encode_prompt(self, prompt: str) -> list[int]:
        """
        Encode a prompt into a list of token IDs.

        ### Parameters
        - prompt: prompt

        ### Returns
        - tokens: list of token IDs
        """
        # aliases
        bots = self.tokenizer.bots
        tokenizer = self.tokenizer.tokenizer

        # tokenization
        tokens = [bots[Actor.user]]
        tokens.extend(tokenizer.encode(prompt))
        tokens.append(bots[Actor.assistant])
        return tokens
