"""
Unit tests for generation.

#### License
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import unittest

import torch
import yaml

from nanollama.model import Transformer, TransformerConfig
from nanollama.utils import initialize_nested_object


class TestGeneration(unittest.TestCase):
    def test_kv_cache_generation(self) -> None:
        bsz = 16
        data = torch.randint(0, 256, (bsz, 1), dtype=torch.long)

        config = yaml.safe_load("""
        vocab_size: 300
        emb_dim: 64
        nb_layers: 2
        block:
            seq_len: 256
            nb_heads: 2
            hidden_dim: 256
        """)

        model = Transformer(initialize_nested_object(TransformerConfig, config))

        # Generation token by token
        model.build_cache(bsz=bsz)

        x = data
        seq_len = x.size(1)

        preds, seq = [], x

        with torch.inference_mode():
            while seq_len <= 256:
                pred = model(x)
                x = pred[:, -1:].argmax(dim=2)
                preds.append(pred)
                seq = torch.concatenate([seq, x], axis=1)
                seq_len += 1

        preds = torch.hstack(preds)

        # Generation all at once
        model.build_cache(bsz=bsz)
        with torch.inference_mode():
            new_data = seq[:, :-1]
            new_preds = model(new_data)

        assert torch.allclose(preds, new_preds, atol=1e-5)
