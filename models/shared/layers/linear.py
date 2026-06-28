from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from models.shared.initialization.weight_init import (
    initialize_linear,
)


class Linear(nn.Module):
    """
    VisionGPT linear projection layer.

    This class intentionally wraps the standard
    linear projection behind a stable interface.

    Future implementations may transparently swap
    the internal implementation for:

    - Tensor Parallel Linear
    - Quantized Linear
    - LoRA Linear
    - Fused CUDA Kernels

    without changing model code.
    """

    __constants__ = (
        "in_features",
        "out_features",
    )

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError(
                "in_features must be greater than zero."
            )

        if out_features <= 0:
            raise ValueError(
                "out_features must be greater than zero."
            )

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features)
            )
        else:
            self.register_parameter(
                "bias",
                None,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize learnable parameters.

        All initialization is delegated to the
        centralized initialization subsystem.
        """

        initialize_linear(
            self.weight,
            self.bias,
        )

    @property
    def bias_enabled(self) -> bool:
        return self.bias is not None

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        inputs

            Shape:

                (..., in_features)

        Returns
        -------
        Tensor

            Shape:

                (..., out_features)
        """

        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension "
                f"{self.in_features}, "
                f"received {inputs.shape[-1]}."
            )

        return F.linear(
            inputs,
            self.weight,
            self.bias,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias_enabled}"
        )