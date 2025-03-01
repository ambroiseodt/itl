"""
Tokenizers based on dialog environment

#### License
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger

import torch

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
    It may be decorated in token space with some `begining of turn` and `end of dialog` tokens.
    """

    source: Actor
    content: str


# ------------------------------------------------------------------------------
# Generic Message Tokenizer
# ------------------------------------------------------------------------------


class Tokenizer(ABC):
    """
    Tokenizer

    ### Attributes
    - name: name of the tokenizer.
    - vocab_size: size of the vocabulary.
    """

    name: str
    vocab_size: int

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


class DialogTokenizer(Tokenizer):
    """
    Dialog Tokenizer

    ### Parameters
    - tokenizer: encoder and decoder from string to list of integers.
    - bots: dictionary mapping actors to `begin_of_turn` tags.
    - eod: token_id to put at the end of a dialog (if `eod != 0`).
    """

    name = "dialog"
    bots: dict[Actor, int]
    eod: bool

    def __init__(self, tokenizer: Tokenizer, bots: dict[Actor, int], eod: int) -> None:
        self.tokenizer = tokenizer
        self.bots = {actor: 0 for actor in Actor} | bots
        self.eod = eod
        self.vocab_size = tokenizer.vocab_size

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
            message = initialize_nested_object(Message, _message, inplace=False)
            bos = self.bots[message.source]
            new_tokens = self.tokenizer.encode(message.content, bos=bos)
            tokens += new_tokens
            if message.source == Actor.assistant:
                mask.extend([True] * len(new_tokens))
            else:
                mask.extend([False] * len(new_tokens))
        if self.eod:
            tokens.append(self.eod)
            mask.append(False)  # do not predict what follows the EoS token
        return tokens, mask

    def decode(self, tokens: list[int], bot_char: str = ":>") -> str:
        """
        Decode a list of token IDs into a sentence.

        ### Parameters
        - tokens: list of token IDs to decode.
        - bot_char: character to signal begining of turn.

        ### Returns
        - decoded sentence.
        """
        output = ""
        for token in tokens:
            output += self.decode_online(token, bot_char=bot_char)
        output += self.flush_decoding_buffer()
        return output

    def decode_online(self, token: int, bot_char: str = ":>") -> str:
        """
        Decode a single token in an online fashion.

        ### Parameters
        - tokens: last token id.
        - bot_char: character to signal begining of turn.

        ### Returns
        - output: decoded sentence.
        """
        # argument parsing
        if isinstance(token, torch.Tensor):
            token = token.item()

        output = ""
        actor = self.bot2actor.get(token, None)

        # if at the start of a new turn, flush the buffer
        if actor:
            add_eol = self.current_actor is not None
            output += self.flush_decoding_buffer()
            output += "\n" if add_eol else ""
            self.current_actor = actor
            output += f"{actor.value.upper()}{bot_char}"

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


class ByteTokenizer(Tokenizer):
    name = "byte"

    error_scheme = "backslashreplace"
    encoding = "utf-8"

    def __init__(self, special_tokens: list[str] = None) -> None:
        """
        Byte Tokenizer

        ### Parameters
        - special_tokens: list of special tokens to register (e.g. `["eos", "bos"]`)
        """
        super().__init__()
        special_tokens = special_tokens if special_tokens else []
        self.vocab_size = 256 + len(special_tokens)
        for i, tok in enumerate(special_tokens):
            logger.info(f"Registering token {tok}")
            setattr(self, tok, i + 256)

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
        - path: path to the tiktoken model.
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

    def encode(self, text: str, bos: int) -> list[int]:
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
    - name: name of the tokenizer.
    - path: path to the tokenizer model.
    - bots: list of actors for which to add `begining of turn` token
    - eod: whether to add an `end of dialog` token.
    """

    name: str
    path: str | None = None
    bots: list[Actor] = field(default_factory=lambda: [actor for actor in Actor])
    eod: bool = True

    def __post_init__(self):
        assert self.name, "Tokenizer name is required."
        self.name = self.name.lower()
        assert self.name in [ByteTokenizer.name, TikTokenTokenizer.name]


def build_tokenizer(config: TokenizerConfig) -> Tokenizer:
    """
    Build tokenizer based on the configuration.

    ### Parameters
    - config: tokenizer configuration.

    ### Returns
    - tokenizer instance.
    """
    special_tokens = [f"bot_{actor.value}" for actor in config.bots]
    if config.eod:
        special_tokens.append("eod")

    if config.name == ByteTokenizer.name:
        tokenizer = ByteTokenizer(special_tokens=special_tokens)

    elif config.name == TikTokenTokenizer.name:
        tokenizer = TikTokenTokenizer(path=config.path)

    else:
        raise NotImplementedError(f"No implementation for tokenizer {config.name}")

    bots: dict[Actor, int] = {actor: getattr(tokenizer, f"bot_{actor.value}") for actor in config.bots}
    eod: int = getattr(tokenizer, "eod", 0)

    return DialogTokenizer(tokenizer=tokenizer, bots=bots, eod=eod)
