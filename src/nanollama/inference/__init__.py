# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Module containing inference utilities.

@ 2025, Meta
"""

from torch import Tensor


def sample_from_logits(logits: Tensor, **kwargs) -> Tensor:
    # TODO implement various sampling strategy
    return logits.argmax(dim=-1)
