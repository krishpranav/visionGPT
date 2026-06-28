from __future__ import annotations

from torch import Tensor
from torch import nn

from models.shared.attention.qkv_projection import (
    QKVProjection,
)
from models.shared.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)
from models.shared.layers.linear import Linear


class MultiHeadAttention(nn.Module):
    """
    VisionGPT Multi-Head Self-Attention.

    Architecture

        Input
          │
          ▼
      QKV Projection
          │
          ▼
    Scaled Dot Product
          │
          ▼
     Output Projection
          │
          ▼
         Output

    Input
    -----

        [B, N, D]

    Output
    ------

        [B, N, D]
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        *,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
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
                "embedding_dim must be divisible by "
                "num_heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.qkv_projection = QKVProjection(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )

        self.attention = ScaledDotProductAttention(
            attention_dropout=attention_dropout,
        )

        self.output_projection = Linear(
            embedding_dim,
            embedding_dim,
        )

        self.output_dropout = nn.Dropout(
            projection_dropout,
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        inputs

            Shape:

                [B, N, D]

        Returns
        -------

        Tensor

            Shape:

                [B, N, D]
        """

        if inputs.ndim != 3:
            raise ValueError(
                "Expected tensor with shape "
                "[B, N, D]."
            )

        batch_size, sequence_length, _ = (
            inputs.shape
        )

        queries, keys, values = (
            self.qkv_projection(inputs)
        )

        outputs = self.attention(
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
        )

        outputs = outputs.transpose(
            1,
            2,
        )

        outputs = outputs.reshape(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        outputs = self.output_projection(
            outputs,
        )

        outputs = self.output_dropout(
            outputs,
        )

        return outputs

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, "
            f"num_heads={self.num_heads}"
        )