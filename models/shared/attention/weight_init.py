from __future__ import annotations
import math
from torch import Tensor
from torch import nn

def initialize_linear(
    weight: Tensor,
    bias: Tensor | None
) -> None:
    if weight.ndim != 2:
        raise ValueError(
            "Linear weight must be 2-D"
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
    std: float = 0.02
) -> None:
    nn.init.trunc_normal_(
        embedding,
        mean = 0.0,
        std = std,
    )

def initialize_rms_norm(
    weight: Tensor
) -> None:
    nn.init.ones_(weight)