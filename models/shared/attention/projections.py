from __future__ import annotations
from torch import Tensor
from torch import nn
from models.shared.layers.linear import Linear


class QKVProjection(nn.Module):
    """
    Projects token embeddings into query, key,
    and value representations.

    Input:

        [B, N, D]

    Output:

        Query
        Key
        Value

    Each output shape:

        [B, H, N, Dh]
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible "
                "by num_heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.head_dim = (
            embedding_dim // num_heads
        )

        self.qkv_projection = Linear(
            embedding_dim,
            embedding_dim * 3,
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if inputs.ndim != 3:
            raise ValueError(
                "Expected tensor with shape "
                "[B, N, D]."
            )

        batch_size, sequence_length, _ = (
            inputs.shape
        )

        qkv = self.qkv_projection(inputs)

        qkv = qkv.view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )

        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4,
        )

        queries = qkv[0]

        keys = qkv[1]

        values = qkv[2]

        return (
            queries,
            keys,
            values,
        )