# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Unit tests for generation.

Copyright (c) 2025 by the authors
"""

# %%
import unittest

import torch
import yaml

from nanollama.model import Transformer, TransformerConfig
from nanollama.model.transformer.architecture import KVCache
from nanollama.utils import build_with_type_check


class TestGeneration(unittest.TestCase):
    def setUp(self) -> None:
        # get some data and a model
        bsz = 8
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data = torch.randint(0, 10, (bsz, 1), dtype=torch.long, device=self.device)
        config = yaml.safe_load("""
        vocab_size: 10
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
        config = build_with_type_check(TransformerConfig, config)
        config.post_init()
        self.model = Transformer(config).to(self.device)

    def test_kv_cache_generation(self) -> None:
        # Generation token by token
        x = self.data

        self.model.set_mode("generate")
        bsz, seq_len = x.size()
        cache_size = (bsz, self.model.nb_kv_heads, 32, self.model.head_dim)
        for layer in self.model.layers:
            layer.attn.kv_cache = KVCache(cache_size, device=self.model.device, dtype=self.model.dtype)

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
        self.model.set_mode("train")
        for layer in self.model.layers:
            layer.attn.kv_cache = None
        with torch.inference_mode():
            new_data = seq[:, :-1]
            new_preds = self.model(new_data)

        assert torch.allclose(preds, new_preds, atol=1e-5)

    def test_parallel_generation(self) -> None:
        # Generation in parallel
        x = self.data

        self.model.set_mode("generate")
        bsz, seq_len = x.size()
        cache_size = (bsz, self.model.nb_kv_heads, 32, self.model.head_dim)
        for layer in self.model.layers:
            layer.attn.kv_cache = KVCache(cache_size, device=self.model.device, dtype=self.model.dtype)

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
            cache_size = (1, self.model.nb_kv_heads, 32, self.model.head_dim)
            for layer in self.model.layers:
                layer.attn.kv_cache = KVCache(cache_size, device=self.model.device, dtype=self.model.dtype)
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
