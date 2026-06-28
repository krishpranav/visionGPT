from __future__ import annotations

import torch
from torch import Tensor
from torch import nn
from enum import StrEnum

from models.shared.initialization.weight_init import (
    initialize_layer_norm,
    initialize_rms_norm,
)


class NormalizationType(StrEnum):
    """
    Supported normalization layers.
    """

    LAYER_NORM = "layer_norm"

    RMS_NORM = "rms_norm"


class LayerNorm(nn.Module):
    """
    VisionGPT Layer Normalization.

    Thin wrapper around ``torch.nn.LayerNorm`` that
    centralizes parameter initialization and exposes
    a stable interface for future optimizations.
    """

    __constants__ = (
        "embedding_dim",
        "eps",
    )

    def __init__(
        self,
        embedding_dim: int,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        self.embedding_dim = embedding_dim
        self.eps = eps

        self.norm = nn.LayerNorm(
            normalized_shape=embedding_dim,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

        if elementwise_affine:
            initialize_layer_norm(
                self.norm.weight,
                self.norm.bias,
            )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        if inputs.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected last dimension "
                f"{self.embedding_dim}, "
                f"received {inputs.shape[-1]}."
            )

        return self.norm(inputs)

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, "
            f"eps={self.eps}"
        )


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Reference:

        Zhang & Sennrich (2019)

    RMSNorm normalizes using only the root mean square
    without subtracting the mean.
    """

    __constants__ = (
        "embedding_dim",
        "eps",
    )

    def __init__(
        self,
        embedding_dim: int,
        *,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        self.embedding_dim = embedding_dim
        self.eps = eps

        self.weight = nn.Parameter(
            torch.empty(embedding_dim)
        )

        initialize_rms_norm(
            self.weight
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        if inputs.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected last dimension "
                f"{self.embedding_dim}, "
                f"received {inputs.shape[-1]}."
            )

        #
        # Compute normalization in FP32 for
        # numerical stability.
        #
        inputs_fp32 = inputs.float()

        rms = torch.rsqrt(
            inputs_fp32.square().mean(
                dim=-1,
                keepdim=True,
            )
            + self.eps
        )

        outputs = (
            inputs_fp32 * rms
        ).to(inputs.dtype)

        return outputs * self.weight

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, "
            f"eps={self.eps}"
        )


def build_normalization(
    normalization_type: NormalizationType,
    embedding_dim: int,
    *,
    eps: float | None = None,
) -> nn.Module:
    """
    Factory used throughout VisionGPT.

    Model code should never instantiate a
    normalization layer directly.
    """

    if normalization_type is NormalizationType.LAYER_NORM:
        return LayerNorm(
            embedding_dim=embedding_dim,
            eps=1e-5 if eps is None else eps,
        )

    if normalization_type is NormalizationType.RMS_NORM:
        return RMSNorm(
            embedding_dim=embedding_dim,
            eps=1e-6 if eps is None else eps,
        )

    raise ValueError(
        f"Unsupported normalization type: "
        f"{normalization_type}"
    )