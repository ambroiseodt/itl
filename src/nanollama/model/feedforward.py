"""
Feed-forward network.

#### License
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
