from __future__ import annotations
from torch import Tensor
from torch import nn


class SwiGLU(nn.Module):
    """
    SwiGLU activation.

    Reference:
        Noam Shazeer, "GLU Variants Improve Transformer"

    Input:
        [..., 2 * hidden_dim]

    Output:
        [..., hidden_dim]

    The input tensor is split into two equal parts:

        value | gate

    Output:

        value * silu(gate)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        if inputs.ndim < 2:
            raise ValueError(
                "Expected tensor with at least two dimensions."
            )

        hidden_size = inputs.shape[-1]

        if hidden_size % 2 != 0:
            raise ValueError(
                "Last dimension must be divisible by 2."
            )

        value, gate = inputs.chunk(
            chunks=2,
            dim=-1,
        )

        return value * nn.functional.silu(gate)