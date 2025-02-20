"""
Tokenizers based on dialog environment

#### License
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta

#### TODO
The dialog decoding is minimalistic.
It should be improved by catching special tokens and adding "\n<{Actor}> " for bos, or "</{Actor}>\n" for eos.
I wrote it offline, I should ask ChatGPT how to do it when back online. If you see this note, is that I forgot to do it.

Moreover, the dialog tokenizer should output a learnable list, to know which token should be learned by the LLM
The LLM loss should only be trained on messages for which the source is `assistant`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger

from ..utils import initialize_nested_object

logger = getLogger("nanollama")


# ------------------------------------------------------------------------------
# Classes for dialog environment
# ------------------------------------------------------------------------------


class Actor(str, Enum):
    """
    Potential interlocutor in a dialog, it could be:
    - a `user` (i.e a human) asking a question
    - an `assistant` (i.e. an LLM) answering

    It may also be tools, in particular:
    - a `database` providing response to a query
    """

    user = "user"
    assistant = "assistant"
    database = "database"


@dataclass
class Message:
    """
    A message is made of a content and a source.
    It may be decorated in token space with some bos and eos tokens.
    """

    source: Actor
    content: str


# ------------------------------------------------------------------------------
# Generic Message Tokenizer
# ------------------------------------------------------------------------------


class Tokenizer(ABC):
    name: str
    vocab_size: int

    @abstractmethod
    def encode(self, sentence: str, bos: int = 0, eos: int = 0) -> list[int]:
        """
        Encode a sentence into a list of token IDs.

        ### Parameters
        sentence: sentence to encode.
        bos: token id to add at the beginning (if bos != 0)
        eos: token id to add at the end (if eos != 0)

        ### Returns
        list of token IDs.
        """
        ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of token IDs into a sentence.

        ### Parameters
        tokens: list of token IDs to decode.

        ### Returns
        decoded sentence.
        """
        ...


# ------------------------------------------------------------------------------
# Dialog Tokenizer
# ------------------------------------------------------------------------------


class DialogTokenizer:
    """
    Dialog Tokenizer

    ### Parameters
    tokenizer: encoder and decoder from string to list of integers.
    bos: dictionary mapping actors to begin_of_sentence tags.
    eos: dictionary mapping actors to end_of_sentence tags.
    """

    def __init__(self, tokenizer: Tokenizer, bos: dict[Actor, int], eos: dict[Actor, int]) -> None:
        self.tokenizer = tokenizer
        self.bos = {actor: 0 for actor in Actor} | bos
        self.eos = {actor: 0 for actor in Actor} | eos

    def encode(self, dialog: list[dict[str, str]]) -> list[int]:
        """
        Encode a dialog into a list of token IDs.

        ### Parameters
        dialog: list of messages

        ### Returns
        list of token IDs.
        """
        tokens = []
        for message in dialog:
            message = initialize_nested_object(Message, message)
            bos = self.bos[message.source]
            eos = self.eos[message.source]
            tokens += self.tokenizer.encode(message.content, bos=bos, eos=eos)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of token IDs into a sentence.

        ### Parameters
        tokens: list of token IDs to decode.

        ### Returns
        decoded sentence.
        """
        import pdb

        pdb.set_trace()
        return self.tokenizer.decode(tokens)


# ------------------------------------------------------------------------------
# Byte Tokenizer
# ------------------------------------------------------------------------------


class ByteTokenizer(Tokenizer):
    name = "byte"

    error_scheme = "backslashreplace"
    encoding = "utf-8"

    def __init__(self, special_tokens: list[str] = None) -> None:
        """
        Byte Tokenizer

        ### Parameters
        special_tokens: list of special tokens to register (e.g. `["eos", "bos"]`)
        """
        super().__init__()
        special_tokens = special_tokens if special_tokens else []
        self.vocab_size = 256 + len(special_tokens)
        for i, tok in enumerate(special_tokens):
            logger.info(f"Registering token {tok}")
            setattr(self, tok, i + 256)

    def encode(self, sentence: str, bos: int = 0, eos: int = 0) -> list[int]:
        tokens = []
        if bos:
            tokens.append(bos)
        tokens.extend(sentence.encode(self.encoding))
        if eos:
            tokens.append(eos)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        byte_tokens = bytes([t for t in tokens if t < 256])
        return byte_tokens.decode(self.encoding, errors=self.error_scheme)


# ------------------------------------------------------------------------------
# Tiktoken Tokenizer
# ------------------------------------------------------------------------------

"""
TODO

I have put some code that I found online, it needs to be modified to fit our needs.
"""
import functools  # noqa: E402
import operator  # noqa: E402
from copy import copy  # noqa: E402
from pathlib import Path  # noqa: E402


class TikTokenTokenizer(Tokenizer):
    name = "tiktoken"

    NUM_RESERVED_TOKENS = 256
    DEFAULT_TIKTOKEN_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa: E501
    DEFAULT_TIKTOKEN_SPECIAL_TOKENS = {
        "<|begin_of_text|>": 0,
        "<|end_of_text|>": 1,
        "<|fim_prefix|>": 2,
        "<|fim_middle|>": 3,
        "<|fim_end_fill|>": 253,
        "<|fim_pad|>": 254,
        "<|fim_suffix|>": 255,
    }
    TIKTOKEN_MAX_ENCODE_CHARS = 400_000

    def __init__(self, path: str) -> None:
        """
        Tiktoken Tokenizer

        ### Parameters
        path: path to the tiktoken model.
        """
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe

        super().__init__()
        mergeable_ranks = load_tiktoken_bpe(path)
        all_special_tokens_with_ids = self.get_all_special_tokens_with_ids(mergeable_ranks)
        self.tkt_model = tiktoken.core.Encoding(
            name=Path(path).stem,
            pat_str=self.DEFAULT_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=all_special_tokens_with_ids,
        )
        self.bos_id: int = self.tkt_model.encode_single_token("<|begin_of_text|>")
        self.eos_id: int = self.tkt_model.encode_single_token("<|end_of_text|>")
        self.n_words: int = self.tkt_model.n_vocab

    def get_all_special_tokens_with_ids(self, mergeable_ranks: dict[bytes, int]) -> dict:
        all_special_tokens_with_ids = copy(self.DEFAULT_TIKTOKEN_SPECIAL_TOKENS)
        missing_ids = set(range(self.NUM_RESERVED_TOKENS)) - set(all_special_tokens_with_ids.values())
        for id_ in missing_ids:
            all_special_tokens_with_ids[f"<|reserved_special_token_{id_}|>"] = id_
        for name in all_special_tokens_with_ids:
            all_special_tokens_with_ids[name] += len(mergeable_ranks)
        return all_special_tokens_with_ids

    def id_to_piece(self, token_id: int) -> str:
        return self.tkt_model.decode_single_token_bytes(token_id).decode()

    def piece_to_id(self, piece: str) -> int:
        return piece

    def encode(self, text: str, bos: int, eos: int) -> list[int]:
        assert isinstance(text, str)
        subs = [
            text[i : i + self.TIKTOKEN_MAX_ENCODE_CHARS] for i in range(0, len(text), self.TIKTOKEN_MAX_ENCODE_CHARS)
        ]
        t: list = functools.reduce(
            operator.iadd,
            self.tkt_model.encode_ordinary_batch(subs),
            [],
        )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, tokens: list[int]) -> str:
        return self.tkt_model.decode(tokens)


# ------------------------------------------------------------------------------
# Main Configuration and Dispatcher
# ------------------------------------------------------------------------------


@dataclass
class TokenizerConfig:
    """
    Tokenizer configuration

    ### Attributes
    name: name of the tokenizer.
    path: path to the tokenizer model.
    bos_actor: list of actors for which to add bos token
    eos_actor: list of actors for which to add eos token
    """

    name: str
    path: str | None = None
    bos_actor: list[Actor] = field(default_factory=lambda: [actor for actor in Actor])
    eos_actor: list[Actor] = field(default_factory=list)

    def __post_init__(self):
        assert self.name, "Tokenizer name is required."
        self.name = self.name.lower()
        assert self.name in [ByteTokenizer.name, TikTokenTokenizer.name]


def build_tokenizer(config: TokenizerConfig) -> Tokenizer:
    """
    Build tokenizer based on the configuration.

    ### Parameters
    config: tokenizer configuration.

    ### Returns
    tokenizer instance.
    """
    bos_tokens = [f"bos_{actor}" for actor in config.bos_actor]
    eos_tokens = [f"eos_{actor}" for actor in config.eos_actor]

    if config.name == ByteTokenizer.name:
        tokenizer = ByteTokenizer(special_tokens=bos_tokens + eos_tokens)

    elif config.name == TikTokenTokenizer.name:
        tokenizer = TikTokenTokenizer(path=config.path)

    else:
        raise NotImplementedError(f"No implementation for tokenizer {config.name}")

    bos = {actor: getattr(tokenizer, f"bos_{actor}") for actor in config.bos_actor}
    eos = {actor: getattr(tokenizer, f"eos_{actor}") for actor in config.eos_actor}

    return DialogTokenizer(tokenizer=tokenizer, bos=bos, eos=eos)
