# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Module to generate completions in parallel with a attention masking mechanism when lengths differs.

@ 2025, Meta

### Notes
This implementation does not support agent calls.
"""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from ..agent import Actor
from ..data.tokenizer import DialogTokenizer
from ..model import Transformer


class MaskedBatchedInference:
    """
    Batched inference with document masking.

    ### Attributes
    - model: transformer model
    - tokenizer: tokenizer
    """

    def __init__(self, model: Transformer, tokenizer: DialogTokenizer):
        # Attributes for generation and decoding
        self.model = model
        self.tokenizer = tokenizer
        self.batch_offset: Tensor

    @property
    def device(self) -> torch.device:
        return self.model.device

    @torch.inference_mode()
    def generate(self, prompts: list[str]) -> list[str]:
        """
        Generate completions for the given prompts.

        ### Parameters
        - prompts: list of prompts
        - gen_len: generation length

        ### Returns
        - output: list of completions
        """
        x = self.build_batch(prompts)
        bsz, total_len = x.size()
        self.model.build_cache(bsz)

        output = []

        while total_len < self.model.seq_len:
            mask = self.build_mask(x.size(1))
            preds = self.model(x, mask=mask)
            x = preds[:, -1:].argmax(dim=-1)
            total_len += 1
            output.append(x)

        output = torch.hstack(output)
        return [self.tokenizer.decode(out.tolist()) for out in output]

    def build_batch(self, prompts: list[str]) -> Tensor:
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
        seq_len = max([len(datum) for datum in data])
        dtype, device = torch.long, self.device

        x = torch.zeros((bsz, seq_len), dtype=dtype, device=device)
        self.batch_offset = torch.zeros(bsz, dtype=torch.long, device=device)
        for i, datum in enumerate(data):
            x[i, -len(datum) :] = datum
            self.batch_offset[i] = seq_len - len(datum)
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

    def build_mask(self, seq_len: int) -> BlockMask:
        """
        Build attention masks for inference, ensuring that attention do not attend to padded tokens.

        ### Parameters
        - seq_len: sequence length currently processed

        ### Attributes
        - mask: attention mask

        ### Notes
        Ideally, we would precomputed a big document mask, and a big causal mask, and simply adjust them.
        However, at time of writting, flex attention does not seem to allow it.
        """
        cur_len = self.model.kv_caches[0].pos_idx
        bsz = len(self.batch_offset)
        seq_len = cur_len + seq_len

        # prefilling
        if cur_len == 0:
            return create_block_mask(self._doc_causal_mask, bsz, None, seq_len, seq_len, device=self.device)

        # online generation
        assert seq_len - cur_len == 1, "only one token can be generated at a time"

        # save overhead if no mask is needed
        if (self.batch_offset == 0).all():
            return None

        return create_block_mask(self._doc_mask, bsz, None, 1, seq_len, device=self.device)

    def _doc_mask(self, b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx >= self.batch_offset[b]

    def _doc_causal_mask(self, b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        return (kv_idx >= self.batch_offset[b]) & (q_idx >= kv_idx)
