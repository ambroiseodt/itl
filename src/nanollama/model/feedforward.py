# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Feed-forward network.

@ 2025, Meta
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ------------------------------------------------------------------------------
# Feed-forward Layer
# ------------------------------------------------------------------------------


class FeedForward(nn.Module):
    """
    Feed-forward network in transformer architecture.

    ### Parameters
    emb_dim: embedding dimension of the inputs
    hidden_dim: hidden dimension of the MLP
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.W_in1 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.W_in2 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        out = F.silu(self.W_in1(x)) * self.W_in2(x)
        out = self.W_out(out)
        return out

    def reset_parameters(self, init_std: float = None, factor: float = 1.0) -> None:
        """Weight initialization"""
        # input
        in_std = init_std or (self.emb_dim ** (-0.5))
        for W_in in [self.W_in1, self.W_in2]:
            nn.init.trunc_normal_(
                W_in.weight,
                mean=0.0,
                std=in_std,
                a=-3 * in_std,
                b=3 * in_std,
            )

        # output
        out_std = init_std or (self.hidden_dim ** (-0.5))
        out_std = out_std / factor
        nn.init.trunc_normal_(
            self.W_out.weight,
            mean=0.0,
            std=out_std,
            a=-3 * out_std,
            b=3 * out_std,
        )

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
        mode_multiplier = dict(fwd=1, bwd=2, both=3)[mode]
        flops = 2 * 3 * self.W_in1.weight.numel()
        return flops * mode_multiplier
