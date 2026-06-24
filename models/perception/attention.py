from __future__ import annotations

import math

import torch
from torch import Tensor
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention.

    Input:

        [B, N, D]

    Output:

        [B, N, D]

    where:

        B = batch size
        N = sequence length
        D = embedding dimension
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
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
                "embedding_dim must be divisible "
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
            bias=True,
        )

        self.attention_dropout = nn.Dropout(
            attention_dropout
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=True,
        )

        self.output_dropout = nn.Dropout(
            projection_dropout
        )

    def forward(
        self,
        tokens: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        tokens

            Shape:
                [B, N, D]

        Returns
        -------
        Tensor

            Shape:
                [B, N, D]
        """

        if tokens.ndim != 3:
            raise ValueError(
                "Expected tensor shape "
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

        attention_scores = (
            queries @ keys.transpose(-2, -1)
        ) * self.scale

        attention_weights = (
            attention_scores.softmax(dim=-1)
        )

        attention_weights = (
            self.attention_dropout(
                attention_weights
            )
        )

        output = (
            attention_weights @ values
        )

        output = output.transpose(
            1,
            2,
        )

        output = output.reshape(
            batch_size,
            num_tokens,
            self.embedding_dim,
        )

        output = self.output_projection(
            output
        )

        output = self.output_dropout(
            output
        )

        return output