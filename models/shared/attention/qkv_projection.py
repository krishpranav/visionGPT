from __future__ import annotations

from torch import Tensor
from torch import nn

from models.shared.layers.linear import Linear


class QKVProjection(nn.Module):
    """
    Projects token embeddings into query, key,
    and value representations.

    Input
    -----
        Tensor
            Shape:
                [B, N, D]

    Output
    ------
        tuple[Tensor, Tensor, Tensor]

        Query:
            [B, H, N, Dh]

        Key:
            [B, H, N, Dh]

        Value:
            [B, H, N, Dh]
    """

    __constants__ = (
        "embedding_dim",
        "num_heads",
        "head_dim",
    )

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be greater than zero."
            )

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

        self.projection = Linear(
            embedding_dim,
            embedding_dim * 3,
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        inputs

            Shape:
                [B, N, D]

        Returns
        -------
        tuple

            Query:
                [B, H, N, Dh]

            Key:
                [B, H, N, Dh]

            Value:
                [B, H, N, Dh]
        """

        if inputs.ndim != 3:
            raise ValueError(
                "Expected tensor with shape "
                "[B, N, D]."
            )

        batch_size, sequence_length, embedding_dim = (
            inputs.shape
        )

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, "
                f"received {embedding_dim}."
            )

        qkv = self.projection(inputs)

        qkv = qkv.reshape(
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

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}"
        )