# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Unit tests for tokenizer.

Copyright (c) 2025 by the authors
"""

import unittest

from nanollama.tokenizer import build_tokenizer


class TestTokenizer(unittest.TestCase):
    def test_dialog_token(self) -> None:
        tokenizer = build_tokenizer({"implementation": "byte", "special_tokens": {"<|user|>": 0, "<|assistant|>": 1}})

        dialog = [
            {"content": "Salut Assistant", "source": "user"},
            {"content": "Bonjour", "source": "assistant"},
            {"content": "Comment vas-tu?", "source": "user"},
            {"content": "Tres bien, comment puis-je vous etre utile aujourd'hui?", "source": "assistant"},
        ]

        tokens = tokenizer.encode(dialog)[0]
        decoded = tokenizer.decode(tokens)
        print(decoded)
        assert (
            decoded
            == "<|user|>Salut Assistant\n<|assistant|>Bonjour\n<|user|>Comment vas-tu?\n<|assistant|>Tres bien, comment puis-je vous etre utile aujourd'hui?"  # noqa: E501
        )
