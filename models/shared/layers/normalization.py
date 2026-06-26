from __future__ import annotations
from enum import StrEnum
import torch
from torch import Tensor
from torch import nn

class NormalizationType(StrEnum):
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"

class LayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        *,
        eps: float = 1e-5,
        elementwise_affline: bool = True,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero"
            )

        self.embedding_dim = embedding_dim
        self.eps = eps

        self.norm = nn.LayerNorm(
            normalized_shape = embedding_dim,
            eps = eps,
            elementwise_affine = elementwise_affline,
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        return self.norm(inputs)

class RMSNorm(nn.Module):
    # todo