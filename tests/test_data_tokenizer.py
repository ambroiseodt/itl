"""
Unit tests for tokenizer.

#### License
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import unittest

from nanollama.data.tokenizer import TokenizerConfig, build_tokenizer


class TestMultipleSourcesTokenGenerator(unittest.TestCase):
    def test_dialog_token(self) -> None:
        config = TokenizerConfig(name="byte")
        tokenizer = build_tokenizer(config)

        dialog = [
            {"content": "Salut Assistant", "source": "user"},
            {"content": "Bonjour", "source": "assistant"},
            {"content": "Comment vas-tu?", "source": "user"},
            {"content": "Tres bien, comment puis-je vous etre utile aujourd'hui?", "source": "assistant"},
        ]

        tokens = tokenizer.encode(dialog)[0]
        decoded = tokenizer.decode(tokens)
        assert decoded == "Salut AssistantBonjourComment vas-tu?Tres bien, comment puis-je vous etre utile aujourd'hui?"
