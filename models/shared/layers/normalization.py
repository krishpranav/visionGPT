from __future__ import annotations
from enum import StrEnum
import torch
from torch import Tensor
from torch import nn


class NormalizationType(StrEnum):
    """
    Supported normalization layers.
    """

    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"


class LayerNorm(nn.Module):
    """
    Wrapper around layer normalization.

    Using our own implementation gives VisionGPT
    a stable interface while allowing future
    optimizations without touching model code.
    """

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
                "embedding_dim must be greater than zero"
            )

        self.embedding_dim = embedding_dim
        self.eps = eps

        self.norm = nn.LayerNorm(
            normalized_shape=embedding_dim,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        return self.norm(inputs)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Reference:

        Root Mean Square Layer Normalization
        Zhang & Sennrich (2019)

    This implementation avoids mean-centering and
    is commonly used in modern large language models.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero"
            )

        self.embedding_dim = embedding_dim
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(embedding_dim)
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        if inputs.shape[-1] != self.embedding_dim:
            raise ValueError(
                "Last dimension of the input tensor "
                f"must be {self.embedding_dim}, "
                f"received {inputs.shape[-1]}"
            )

        inputs_fp32 = inputs.float()

        rms = torch.rsqrt(
            inputs_fp32.pow(2).mean(
                dim=-1,
                keepdim=True,
            )
            + self.eps
        )

        outputs = inputs_fp32 * rms

        outputs = outputs.to(inputs.dtype)

        return outputs * self.weight


def build_normalization(
    normalization_type: NormalizationType,
    embedding_dim: int,
    *,
    eps: float | None = None,
) -> nn.Module:
    """
    Factory function used throughout VisionGPT.

    This centralizes normalization creation so
    model code never depends on specific
    normalization implementations.
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