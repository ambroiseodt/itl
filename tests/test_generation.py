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
        bsz = 8
        self.data = [torch.randint(0, 30, (1,), dtype=torch.long, device="cuda") for _ in range(bsz)]
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
        self.model = Transformer(initialize_nested_object(TransformerConfig, config)).to("cuda")

    def test_kv_cache_generation(self) -> None:

        # Generation token by token
        x = self.model.setup_inference(self.data)
        seq_len = x.size(1)
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
        self.model.setup_training()
        with torch.inference_mode():
            new_data = seq[:, :-1]
            new_preds = self.model(new_data)

        assert torch.allclose(preds, new_preds, atol=1e-5)

    def test_parallel_generation(self) -> None:
        # Generation in parallel
        x = self.model.setup_inference(self.data)
        seq_len = x.size(1)
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
        for data in self.data:
            x = self.model.setup_inference([data])
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

    def test_shifted_inputs(self) -> None:
        torch.cuda.manual_seed(0)

        bsz = len(self.data)
        offset = 10
        _data = torch.randint(0, 30, (bsz, 20), dtype=torch.long, device="cuda")
        prefix_lens = torch.randint(offset, 20, (bsz,), dtype=torch.long, device="cuda")
        data = [datum[:dlen] for datum, dlen in zip(_data, prefix_lens)]

        # shift data
        x = self.model.setup_inference(data)
        doc_start = self.model.batch_offset
        tmp = torch.arange(bsz, device="cuda")
        assert torch.allclose(x[tmp, doc_start], _data[:, 0])

        # completion of all prompts in parallel with shifted inputs
        gen_len = 12
        with torch.inference_mode():
            # prefilling
            pred_ = self.model(x)

            # online generation
            pred = pred_
            buffer = []
            seq_len = 0
            while seq_len < gen_len:
                token = pred[:, -1:].argmax(dim=2)
                seq_len += 1
                buffer.append(token)
                pred = self.model(token)
        seq = torch.hstack(buffer).tolist()[0]

        # completion of all prompts in parallel without shifted inputs
        x = _data[:, :offset]
        self.model.setup_training()
        with torch.inference_mode():
            new_pred = self.model(x)

        # realign prefilling predictions to compare them
        pred = torch.empty((bsz, offset, 30), dtype=pred_.dtype, device="cuda")
        for i in range(bsz):
            start = doc_start[i].item()
            pred[i] = pred_[i, start : start + offset]

        assert torch.allclose(pred, new_pred, atol=1e-5)

        # online generation with a single prompt
        # prefilling
        prompts = [data[0]]
        x = self.model.setup_inference(prompts)
        pred = self.model(x)

        # generation
        seq_len = 0
        buffer = []
        with torch.inference_mode():
            while seq_len < gen_len:
                token = pred[:, -1:].argmax(dim=2)
                seq_len += 1
                buffer.append(token)
                pred = self.model(token)
        ref_seq = torch.hstack(buffer).tolist()[0]

        assert seq == ref_seq
