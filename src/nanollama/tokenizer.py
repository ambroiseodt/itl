# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Module providing tokenizers to cast dialog environments into lists of tokens

@ 2025, Meta
"""

import functools
import json
import operator
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, Literal

from torch import Tensor

from .agent import Actor
from .utils import build_with_type_check

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# Generic Message Tokenizer
# ------------------------------------------------------------------------------


class Tokenizer(ABC):
    """
    ### Attributes
    - vocab_size: size of the vocabulary.
    """

    vocab_size: int

    def _register_special_tokens(self, special_tokens: dict[str, int], offset: int) -> None:
        """
        Add attributes to the tokenizer corresponding to the special tokens

        ### Parameters
        - special_tokens: dictionary of special tokens to their ids
        - offset: offset to avoid collision with existing tokens
        """
        logger.debug(f"Registering special tokens: {special_tokens}")

        for tok_str, tok_id in special_tokens.items():
            tok = re.sub(r"<\|(.+?)\|>", r"\1", tok_str)  # Remove <| and |>
            setattr(self, tok, tok_id + offset)

    @abstractmethod
    def encode(self, sentence: str, bos: int = 0) -> list[int]:
        """
        Encode a sentence into a list of token IDs.

        ### Parameters
        - sentence: sentence to encode.
        - bos: token id to add at the beginning of sentence (if `bos != 0`)

        ### Returns
        - tokens: list of token IDs.
        """
        ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of token IDs into a sentence.

        ### Parameters
        - tokens: list of token IDs to decode.

        ### Returns
        - decoded sentence.
        """
        ...


# ------------------------------------------------------------------------------
# Dialog Tokenizer
# ------------------------------------------------------------------------------


@dataclass
class Message:
    """
    A dialog is made of a list of messages.
    A message is made of a content and a source.

    ### Parameters
    - source: the actor that produced the message.
    - content: the content of the message.
    """

    source: Actor
    content: str


class DialogTokenizer:
    """
    ### Parameters
    - tokenizer: encoder and decoder from string to list of integers.

    ### Attributes
    - bots: dictionary mapping actors to `begin_of_turn` tags.
    - eod: token_id to put at the end of a dialog (if `eod != 0`).
    """

    bots: dict[Actor, int]
    eod: int

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        # attributes associated to dialog format
        self.bots = {actor: getattr(self.tokenizer, actor.value, 0) for actor in Actor}
        self.eod = getattr(self.tokenizer, "eod", 0)

        # decoding attributes
        self.bot2actor = {v: k for k, v in self.bots.items()}
        if 0 in self.bot2actor:
            self.bot2actor.pop(0)
        self.current_actor = None
        self.buffer = []

    def encode(self, dialog: list[dict[str, str]]) -> tuple[list[int], list[bool]]:
        """
        Encode a dialog into a list of token IDs.

        ### Parameters
        - dialog: list of messages

        ### Returns
        - tokens: list of token IDs
        - mask: list of boolean flag specifying tokens produced by the assistant

        ### Note
        The mask specifies tokens for which the LLM needs to predict the next token.
        This is because each turn starts with a special `bot` tokens.
        When we end a non-assistant turn, we manually add the assistant bos token.
        When the assistant finishes its turn, it chooses the next actor by predicting its `bos` token.
        """
        tokens = []
        mask = []
        for _message in dialog:
            message = build_with_type_check(Message, _message, inplace=False)
            bos = self.bots[message.source]
            new_tokens = self.tokenizer.encode(message.content, bos=bos)
            tokens.extend(new_tokens)
            if message.source == Actor.assistant:
                mask.extend([True] * len(new_tokens))
            else:
                mask.extend([False] * len(new_tokens))
        if self.eod:
            tokens.append(self.eod)
            mask.append(False)  # do not predict what follows the EoS token
        return tokens, mask

    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of token IDs into a sentence.

        ### Parameters
        - tokens: list of token IDs to decode.

        ### Returns
        - decoded sentence.
        """
        output = ""
        for token in tokens:
            output += self.decode_online(token)
            if self.eod and token == self.eod:
                break
        output += self.flush_decoding_buffer()
        return output

    def decode_online(self, token: int) -> str:
        """
        Decode a single token in an online fashion.

        ### Parameters
        - tokens: last token id.

        ### Returns
        - output: decoded sentence.
        """
        # argument parsing
        if isinstance(token, Tensor):
            token = token.item()

        output = ""
        actor = self.bot2actor.get(token, None)

        # if at the start of a new turn, flush the buffer
        if actor:
            add_eol = self.current_actor is not None
            output += self.flush_decoding_buffer()
            output += "\n" if add_eol else ""
            self.current_actor = actor
            output += f"<|{actor.value}|>"

        # else bufferize the current token
        else:
            self.buffer.append(token)
        return output

    def flush_decoding_buffer(self) -> str:
        """
        Decode bufferized tokens into text.
        """
        output = self.tokenizer.decode(self.buffer)
        self.current_actor = None
        self.buffer = []
        return output


# ------------------------------------------------------------------------------
# Byte Tokenizer
# ------------------------------------------------------------------------------


@dataclass
class ByteTokenizerConfig:
    """
    Tokenizer configuration

    ### Attributes
    - name: name of the tokenizer.
    - special_tokens: list of special tokens to register (e.g. `["<|eos|>", "<|user|>"]`)
    """

    implementation: Literal["byte"] = "byte"
    special_tokens: dict[str, int] = field(default_factory=dict)


class ByteTokenizer(Tokenizer):
    error_scheme = "backslashreplace"
    encoding = "utf-8"

    def __init__(self, config: ByteTokenizerConfig) -> None:
        """
        Byte Tokenizer

        ### Parameters
        - special_tokens: list of special tokens to register (e.g. `["eos", "bos"]`)
        """
        super().__init__()

        # build special_tokens
        self._register_special_tokens(config.special_tokens, offset=256)

        # register vocabulary size
        self.vocab_size = 256 + max(config.special_tokens.values())

    def encode(self, sentence: str, bos: int = 0) -> list[int]:
        tokens = []
        if bos:
            tokens.append(bos)
        tokens.extend(sentence.encode(self.encoding))
        return tokens

    def decode(self, tokens: list[int]) -> str:
        byte_tokens = bytes([t for t in tokens if t < 256])
        return byte_tokens.decode(self.encoding, errors=self.error_scheme)


# ------------------------------------------------------------------------------
# Tiktoken Tokenizer
# ------------------------------------------------------------------------------


@dataclass
class TikTokenizerConfig:
    """
    TikTokenizer configuration

    ### Attributes
    - name: name of the tokenizer.
    - path: path to the tokenizer model.
    - special_tokens: list of special tokens to register (e.g. `["<|eos|>", "<|user|>"]`)
    - pattern: regex pattern to match special tokens
    - nb_special_tokens: number of special tokens
    """

    path: str
    implementation: Literal["tiktoken"] = "tiktoken"
    special_tokens: dict[str, int] = field(default_factory=dict)
    pattern: str = None
    nb_special_tokens: int = None

    def __post_init__(self) -> None:
        self.path = os.path.expandvars(self.path)
        if self.nb_special_tokens is None:
            self.nb_special_tokens = max(self.special_tokens.values()) + 1

        missing_ids = set(range(self.nb_special_tokens)) - set(self.special_tokens.values())
        for tok_id in missing_ids:
            self.special_tokens[f"<|special_token_{tok_id}|>"] = tok_id


class TikTokenizer(Tokenizer):
    """
    Tiktoken Tokenizer

    ### Parameters
    - tokenizer_dir: directory containing a `merge.bpe` and `params.json` file defining BPE merges, and special tokens.

    The special tokens configuration files should be structured as
    ```json
    nb_special_tokens: ...
    pattern: <regex_pattern>
    special_tokens:
        <token_str>: <token_id>
    ```

    The tokenizer string should be <|eod|>, or of the form <|actor|> where `actor` can be `assistant`, `user`, ...
    """

    ENCODE_CHUNK = 400_000

    def __init__(
        self,
        config: TikTokenizerConfig,
    ) -> None:
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe

        # load merges
        merges = load_tiktoken_bpe(config.path)

        # build special tokens
        # ... add special tokens to restricted tokens slots
        # ... offset all special tokens to restricted tokens slots
        self._register_special_tokens(config.special_tokens, offset=len(merges))

        # build tiktoken engine
        self.engine = tiktoken.core.Encoding(
            name=Path(config.path).name,
            pat_str=config.pattern,
            mergeable_ranks=merges,
            special_tokens=config.special_tokens,
        )

        # register vocabulary size
        self.vocab_size: int = self.engine.n_vocab

    def encode(self, sentence: str, bos: int) -> list[int]:
        subs = [sentence[i : i + self.ENCODE_CHUNK] for i in range(0, len(sentence), self.ENCODE_CHUNK)]
        t: list = functools.reduce(
            operator.iadd,
            self.engine.encode_ordinary_batch(subs),
            [],
        )
        if bos:
            t.insert(0, bos)
        return t

    def decode(self, tokens: list[int]) -> str:
        return self.engine.decode(tokens)

    @staticmethod
    def download_model(name: str, save_dir: str) -> None:
        """
        Download tokenizer from tiktoken repository.

        ### Parameters
        - name: name of the tokenizer to download
        - save_dir: directory to save the tokenizer

        ### Usage
        ```python
        from nanollama.data.tokenizer import TikTokenizer

        TikTokenizer.download_model("cl100k_base", "$HOME/tokenizer")
        ```
        """
        import tiktoken
        from tiktoken.load import dump_tiktoken_bpe

        save_dir = Path(os.path.expandvars(save_dir))
        save_dir.mkdir(parents=True, exist_ok=True)

        # build tiktoken engine
        engine = tiktoken.get_encoding(name)

        # save merges
        merges = engine._mergeable_ranks
        dump_tiktoken_bpe(merges, save_dir / f"{name}.bpe")

        # save parameters
        special_tokens = {}
        specials = engine.special_tokens_set
        for token in specials:
            special_tokens |= {token: engine.encode(token, allowed_special=specials)}
        params = {
            "pattern": engine._pat_str,
            "special_tokens": special_tokens,
            "nb_special_tokens": engine.n_vocab - len(merges),
        }
        with open(save_dir / f"{name}.json", "w") as f:
            print(json.dumps(params), file=f, flush=True)


# ------------------------------------------------------------------------------
# Main Configuration and Dispatcher
# ------------------------------------------------------------------------------


def build_tokenizer(config: dict[str, Any]) -> DialogTokenizer:
    """
    Initialize configuration based on the specified model implementation.

    ### Parameters
    - config: A dictionary containing the configuration details.

    ### Returns
    -
    """
    implementation = config.get("implementation", "byte").lower()

    match implementation:
        case "byte":
            config = build_with_type_check(ByteTokenizerConfig, config)
            tokenizer = ByteTokenizer(config)

        case "tiktoken":
            config = build_with_type_check(TikTokenizerConfig, config)
            tokenizer = TikTokenizer(config)

        case _:
            raise ValueError(f"Tokenizer implementation {implementation} not found")

    return DialogTokenizer(tokenizer=tokenizer)
