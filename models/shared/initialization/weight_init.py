from __future__ import annotations
import math
from torch import Tensor
from torch import nn


def initialize_linear(
    weight: Tensor,
    bias: Tensor | None,
) -> None:
    """
    Initialize a linear layer.

    Current strategy:
        Xavier Uniform

    Bias:
        Uniform(-1/sqrt(fan_in), +1/sqrt(fan_in))

    This is the single source of truth for
    linear initialization throughout VisionGPT.
    """

    if weight.ndim != 2:
        raise ValueError(
            "Linear weight must be 2-dimensional."
        )

    nn.init.xavier_uniform_(weight)

    if bias is None:
        return

    fan_in = weight.shape[1]

    bound = 1.0 / math.sqrt(fan_in)

    nn.init.uniform_(
        bias,
        -bound,
        bound,
    )


def initialize_embedding(
    embedding: Tensor,
    *,
    std: float = 0.02,
) -> None:
    """
    Initialize embedding matrices.

    Used for:

    - Position embeddings
    - Learnable tokens
    - Future token embeddings
    """

    nn.init.trunc_normal_(
        embedding,
        mean=0.0,
        std=std,
    )


def initialize_layer_norm(
    weight: Tensor,
    bias: Tensor | None,
) -> None:
    """
    Initialize LayerNorm parameters.
    """

    nn.init.ones_(weight)

    if bias is not None:
        nn.init.zeros_(bias)


def initialize_rms_norm(
    weight: Tensor,
) -> None:
    """
    Initialize RMSNorm scale parameter.
    """

    nn.init.ones_(weight)