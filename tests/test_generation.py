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

from nanollama.model import Transformer, TransformerConfig
from nanollama.utils import initialize_nested_object


class TestGeneration(unittest.TestCase):
    def setUp(self) -> None:
        # get some data and a model
        bsz = 16
        self.data = torch.randint(0, 256, (bsz, 1), dtype=torch.long)
        config = yaml.safe_load("""
        vocab_size: 300
        emb_dim: 64
        nb_layers: 2
        block:
            seq_len: 256
            nb_heads: 2
            hidden_dim: 256
        """)
        torch.manual_seed(0)
        self.model = Transformer(initialize_nested_object(TransformerConfig, config))

    def test_kv_cache_generation(self) -> None:

        # Generation token by token
        bsz = self.data.size(0)
        self.model.build_cache(bsz=bsz)
        x = self.data
        seq_len = x.size(1)
        preds, seq = [], x
        with torch.inference_mode():
            while seq_len <= 256:
                pred = self.model(x)
                x = pred[:, -1:].argmax(dim=2)
                preds.append(pred)
                seq = torch.concatenate([seq, x], axis=1)
                seq_len += 1
        preds = torch.hstack(preds)

        # Generation all at once
        self.model.build_cache(bsz=bsz)
        with torch.inference_mode():
            new_data = seq[:, :-1]
            new_preds = self.model(new_data)

        assert torch.allclose(preds, new_preds, atol=1e-5)

    def test_parallel_generation(self) -> None:
        # Generation in parallel
        bsz = self.data.size(0)
        self.model.build_cache(bsz=bsz)
        x = self.data
        seq_len = x.size(1)
        preds = []
        with torch.inference_mode():
            while seq_len <= 256:
                pred = self.model(x)
                x = pred[:, -1:].argmax(dim=2)
                preds.append(pred)
                seq_len += 1
        preds = torch.hstack(preds)

        # Generation one sentence at the time
        self.model.build_cache(bsz=1)
        new_preds = []
        for i in range(bsz):
            x = self.data[i : i + 1]
            seq_len = x.size(1)
            new_pred = []
            with torch.inference_mode():
                while seq_len <= 256:
                    pred = self.model(x)
                    x = pred[:, -1:].argmax(dim=2)
                    new_pred.append(pred)
                    seq_len += 1
            new_pred = torch.hstack(new_pred)
            new_preds.append(new_pred)
            self.model.reset_cache()
        new_preds = torch.vstack(new_preds)

        assert torch.allclose(preds, new_preds, atol=1e-5)

    def test_shifted_inputs(self) -> None:
        self.model = self.model.to("cuda")
        self.data = self.data.to("cuda")

        bsz = self.data.size(0)
        offset = 10
        data = torch.randint(0, 256, (bsz, 100), dtype=torch.long, device="cuda")
        prefix_lens = torch.randint(offset, 100, (bsz,), dtype=torch.long, device="cuda")

        # shift data
        shifted_data, doc_start = self.model.build_prompts(data, prefix_lens)
        tmp = torch.arange(bsz, device="cuda")
        assert torch.allclose(shifted_data[tmp, doc_start], data[:, 0])

        # completion of all prompts in parallel with shifted inputs
        x = shifted_data
        self.model.build_cache(bsz=bsz)
        with torch.inference_mode():
            pred_ = self.model(x, doc_start=doc_start)

        # realign the predictions
        pred = torch.empty((bsz, offset, 300), dtype=pred_.dtype, device="cuda")
        for i in range(bsz):
            start = doc_start[i].item()
            pred[i] = pred_[i, start : start + offset]

        # completion of all prompts in parallel without shifted inputs
        x = data[:, :offset]
        self.model.build_cache(bsz=bsz)
        with torch.inference_mode():
            new_pred = self.model(x)

        assert torch.allclose(pred, new_pred, atol=1e-4)
