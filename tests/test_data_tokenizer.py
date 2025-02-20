"""
Unit tests for tokenizer.

#### License
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import unittest

from nanollama.data.tokenizer import Actor, Message, TokenizerConfig, build_tokenizer


class TestMultipleSourcesTokenGenerator(unittest.TestCase):
    def test_dialog_token(self) -> None:
        config = TokenizerConfig(name="byte")
        tokenizer = build_tokenizer(config)

        dialog = [
            Message(content="Salut Assistant", source=Actor.user),
            Message(content="Bonjour", source=Actor.assistant),
            Message(content="Comment vas-tu?", source=Actor.user),
            Message(content="Tres bien, comment puis-je vous etre utile aujourd'hui?", source=Actor.assistant),
        ]

        tokens = tokenizer.encode(dialog)
        decoded = tokenizer.decode(tokens)
        assert decoded == "Salut AssistantBonjourComment vas-tu?Tres bien, comment puis-je vous etre utile aujourd'hui?"
