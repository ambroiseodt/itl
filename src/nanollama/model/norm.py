# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Normalization layers

@ 2025, Ambroise Odonnat
"""

import torch
import torch.nn as nn
from torch import Tensor

# ------------------------------------------------------------------------------
# Normalization Layer
# ------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    RMS normalization layer.

    Parameters
    ----------
        dim:
            dimension of the input tensor
        eps:
            numerical stability parameter
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        output = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)
        return (output * self.weight).type_as(x)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def get_nb_flop(self, mode: str = "both") -> int:
        """
        Number of flop to process a new token

        ### Notes
        The following formula is somewhat sketchy, it would be nice to validate it empirically.
        This could be done by using the torch dispatcher
        https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505

        ### Parameters
        - mode: whether to consider the forward, backward pass or both
        """
        mode_multiplier = dict(fwd=1, bwd=1, both=2)[mode]
        flops = 2 * self.weight.numel()
        return mode_multiplier * flops
