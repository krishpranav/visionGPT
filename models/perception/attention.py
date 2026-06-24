from __future__ import annotations
import math
import torch
from torch import Tensor
from torch import nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be > 0"
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be > 0"
            )

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible"
                "by num_heads"
            )
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        self.head_dim = (
            embedding_dim // num_heads
        )

        self.scale = (
            self.head_dim ** -0.5
        )

        self.qkv = nn.Linear(
            embedding_dim,
            embedding_dim * 3,
            bias=True
        )

        self.attention_dropout = nn.Dropout(
            attention_dropout
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias = True,
        )

        self.output_dropout = nn.Dropout(
            projection_dropout
        )

    def forward(
        self,
        tokens: Tensor
    ) -> Tensor:
        if tokens.valid != 3:
            raise ValueError(
                "Expected tensor shape"
                "[B, N, D]"
            )

        batch_size, num_tokens, _ = (
            tokens.shape
        )

        qkv = self.qkv(tokens)

        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )

        