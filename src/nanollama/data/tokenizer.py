"""
Tokenizers

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

# ------------------------------------------------------------------------------
# Generic Class
# ------------------------------------------------------------------------------


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, sentence: str) -> list[int]:
        """
        Encode a sentence into a list of token IDs.

        Parameters
        ----------
        sentence: sentence to encode.

        Returns
        -------
        list of token IDs.
        """
        ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of token IDs into a sentence.

        Parameters
        ----------
        tokens: list of token IDs to decode.

        Returns
        -------
        decoded sentence.
        """
        ...


# ------------------------------------------------------------------------------
# Byte Tokenizer
# ------------------------------------------------------------------------------


class ByteTokenizer(Tokenizer):
    error_scheme = "backslashreplace"
    encoding = "utf-8"

    def __init__(self, bos: bool = False, eos: bool = False):
        """
        Byte Tokenizer

        Parameters
        ----------
        bos: whether to add a BOS token at the beginning.
        eos: whether to add an EOS token at the end.
        """
        self.bos = bos
        self.eos = eos

        self.vocab_size = 256
        if bos:
            self.bos_id = 256
            self.vocab_size += 1
        if eos:
            self.eos_id = 257
            self.vocab_size += 1

    def encode(self, sentence: str) -> list[int]:
        tokens = []
        if self.bos:
            tokens.append(self.bos_id)
        tokens.extend(sentence.encode(self.encoding))
        if self.eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        byte_tokens = bytes([t for t in tokens if t < 256])
        return byte_tokens.decode(self.encoding, errors=self.error_scheme)


# ------------------------------------------------------------------------------
# Tiktoken Tokenizer
# ------------------------------------------------------------------------------

"""
TODO

Use tiktoken to parse <TOOLUSE> and </TOOLUSE> as special tokens.
Also create a special <BOS> and <EOS> token.

I have put some code that I found online, it needs to be modified to fit our needs.
"""
import functools  # noqa: E402
import operator  # noqa: E402
from copy import copy  # noqa: E402
from pathlib import Path  # noqa: E402


class TikTokenTokenizer(Tokenizer):
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

    def __init__(self, model_path: str, bos: bool = True, eos: bool = False) -> None:
        """
        Tiktoken Tokenizer

        Parameters
        ----------
        model_path: path to the tiktoken model.
        bos: whether to add a BOS token at the beginning.
        eos: whether to add an EOS token at the end.
        """
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe

        super().__init__()
        mergeable_ranks = load_tiktoken_bpe(model_path)
        all_special_tokens_with_ids = self.get_all_special_tokens_with_ids(mergeable_ranks)
        self.tkt_model = tiktoken.core.Encoding(
            name=Path(model_path).stem,
            pat_str=self.DEFAULT_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=all_special_tokens_with_ids,
        )
        self.bos = bos
        self.eos = eos
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

    def encode(self, text: str) -> list[int]:
        assert isinstance(text, str)
        subs = [
            text[i : i + self.TIKTOKEN_MAX_ENCODE_CHARS] for i in range(0, len(text), self.TIKTOKEN_MAX_ENCODE_CHARS)
        ]
        t: list = functools.reduce(
            operator.iadd,
            self.tkt_model.encode_ordinary_batch(subs),
            [],
        )
        if self.bos:
            t.insert(0, self.bos_id)
        if self.eos:
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
    TODO
    """

    name: str
    pass


def build_tokenizer(config: TokenizerConfig) -> Tokenizer:
    """
    TODO: look at Amaia dispatcher
    """
    pass
