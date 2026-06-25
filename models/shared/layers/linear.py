from __future__ import annotations
import math
import torch
from torch import Tensor
from torch import nn


class Linear(nn.Module):
    """
    Standard dense projection used throughout VisionGPT.

    This layer intentionally wraps ``torch.nn.Linear`` to provide
    a stable abstraction for future optimizations such as:

    - Tensor Parallel Linear
    - Quantized Linear
    - LoRA adapters
    - Fused CUDA kernels

    without requiring changes to model code.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError("in_features must be greater than zero")

        if out_features <= 0:
            raise ValueError("out_features must be greater than zero")

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Xavier uniform initialization.

        This is intentionally centralized so future initialization
        strategies can be introduced without changing every model.
        """

        nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        inputs:
            Tensor of shape (..., in_features)

        Returns
        -------
        Tensor
            Tensor of shape (..., out_features)
        """

        return torch.nn.functional.linear(
            inputs,
            self.weight,
            self.bias,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )