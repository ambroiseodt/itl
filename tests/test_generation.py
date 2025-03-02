"""
Unit tests for generation.

#### License
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

# %%
import unittest

import torch
import yaml

from nanollama.inference import MaskedBatchedInference
from nanollama.model import Transformer, TransformerConfig
from nanollama.utils import initialize_nested_object


class TestGeneration(unittest.TestCase):
    def setUp(self) -> None:
        # get some data and a model
        bsz = 8
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data = torch.randint(0, 30, (bsz, 1), dtype=torch.long, device=self.device)
        config = yaml.safe_load("""
        vocab_size: 30
        emb_dim: 8
        nb_layers: 2
        block:
            seq_len: 32
            nb_heads: 2
            hidden_dim: 16
        """)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        self.model = Transformer(initialize_nested_object(TransformerConfig, config)).to(self.device)

    def test_kv_cache_generation(self) -> None:

        # Generation token by token
        x = self.data
        bsz, seq_len = x.size()
        self.model.build_cache(bsz)
        preds, seq = [], x
        with torch.inference_mode():
            while seq_len <= 32:
                pred = self.model(x)
                x = pred[:, -1:].argmax(dim=2)
                preds.append(pred)
                seq = torch.concatenate([seq, x], axis=1)
                seq_len += 1
        preds = torch.hstack(preds)

        # Generation all at once
        self.model.delete_kv_cache()
        with torch.inference_mode():
            new_data = seq[:, :-1]
            new_preds = self.model(new_data)

        assert torch.allclose(preds, new_preds, atol=1e-5)

    def test_parallel_generation(self) -> None:
        # Generation in parallel
        x = self.data
        bsz, seq_len = x.size()
        self.model.build_cache(bsz)
        preds = []
        with torch.inference_mode():
            while seq_len <= 32:
                pred = self.model(x)
                x = pred[:, -1:].argmax(dim=2)
                preds.append(pred)
                seq_len += 1
        preds = torch.hstack(preds)

        # Generation one sentence at the time
        new_preds = []
        for x in self.data:
            x = x.unsqueeze(0)
            self.model.build_cache(1)
            seq_len = x.size(1)
            new_pred = []
            with torch.inference_mode():
                while seq_len <= 32:
                    pred = self.model(x)
                    x = pred[:, -1:].argmax(dim=2)
                    new_pred.append(pred)
                    seq_len += 1
            new_pred = torch.hstack(new_pred)
            new_preds.append(new_pred)
        new_preds = torch.vstack(new_preds)

        assert torch.allclose(preds, new_preds, atol=1e-5)

    def test_mask_batch_inference(self) -> None:
        # hyperparameters
        bsz = len(self.data)
        max_prompt_len = 20
        min_prompt_len = 10
        gen_len = self.model.seq_len - max_prompt_len
        device = self.device

        class MockTokenizer:
            def decode(self, x):
                return x

        inference_engine = MaskedBatchedInference(model=self.model, tokenizer=MockTokenizer())
        # Mock tokenizer effect
        inference_engine.encode_prompt = lambda x: x

        data = torch.randint(0, 30, (bsz, max_prompt_len), dtype=torch.long, device=device)
        prefix_lens = torch.randint(min_prompt_len, max_prompt_len, (bsz,), dtype=torch.long, device=device)
        prompts = [datum[:dlen] for datum, dlen in zip(data, prefix_lens)]

        # check for correct shifting
        x = inference_engine.build_batch(prompts)
        doc_start = inference_engine.batch_offset
        tmp = torch.arange(bsz, device=device)
        assert torch.allclose(x[tmp, doc_start], data[:, 0])

        # completion in parallel or one by one
        completion = inference_engine.generate(prompts)
        ref_completion = [inference_engine.generate([prompt])[0] for prompt in prompts]
        for c, rc in zip(completion, ref_completion):
            assert c == rc[: len(c)]

        ref_completion = inference_engine.generate([prompts[0]])[0]

        # without batch inference
        self.model.build_cache(1)
        x = data[:1, : prefix_lens[0]]
        seq = []
        with torch.inference_mode():
            while len(seq) < len(ref_completion):
                pred = self.model(x)
                x = pred[:, -1:].argmax(dim=2)
                seq.append(x.item())

        assert seq == ref_completion
