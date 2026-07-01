from __future__ import annotations

from torch import Tensor
from torch import nn

from models.shared.attention.multi_head_attention import (
    MultiHeadAttention,
)
from models.shared.layers.mlp import (
    FeedForwardNetwork,
)
from models.shared.layers.normalization import (
    RMSNorm,
)


class TransformerEncoderBlock(nn.Module):
    """
    VisionGPT Transformer Encoder Block.

    Architecture

        Input
          │
          ▼
       RMSNorm
          │
          ▼
    Multi-Head Attention
          │
          ▼
       Residual Add
          │
          ▼
       RMSNorm
          │
          ▼
    Feed Forward Network
          │
          ▼
       Residual Add
          │
          ▼
         Output

    This implements the standard
    pre-normalization transformer block.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dimension: int,
        *,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.attention_norm = RMSNorm(
            embedding_dim,
        )

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )

        self.mlp_norm = RMSNorm(
            embedding_dim,
        )

        self.feed_forward = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            hidden_dimension=hidden_dimension,
            dropout=dropout,
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

        #
        # Pre-Norm Attention
        #
        attention_output = self.attention(
            self.attention_norm(inputs),
            attention_mask=attention_mask,
        )

        outputs = inputs + attention_output

        #
        # Pre-Norm Feed Forward
        #
        mlp_output = self.feed_forward(
            self.mlp_norm(outputs),
        )

        outputs = outputs + mlp_output

        return outputs