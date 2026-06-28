from __future__ import annotations

import math

import torch
from torch import Tensor
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """
    Computes scaled dot-product attention.

    Inputs
    ------
    queries : Tensor
        Shape: [B, H, N, Dh]

    keys : Tensor
        Shape: [B, H, N, Dh]

    values : Tensor
        Shape: [B, H, N, Dh]

    Returns
    -------
    Tensor
        Shape: [B, H, N, Dh]
    """

    def __init__(
        self,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.attention_dropout = nn.Dropout(
            attention_dropout
        )

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Compute scaled dot-product attention.
        """

        if (
            queries.ndim != 4
            or keys.ndim != 4
            or values.ndim != 4
        ):
            raise ValueError(
                "Expected tensors with shape "
                "[B, H, N, Dh]."
            )

        if queries.shape != keys.shape:
            raise ValueError(
                "Query and Key tensors "
                "must have identical shapes."
            )

        if queries.shape != values.shape:
            raise ValueError(
                "Query and Value tensors "
                "must have identical shapes."
            )

        head_dim = queries.shape[-1]

        scale = 1.0 / math.sqrt(head_dim)

        attention_scores = (
            queries @ keys.transpose(-2, -1)
        ) * scale

        if attention_mask is not None:
            attention_scores = (
                attention_scores.masked_fill(
                    attention_mask,
                    float("-inf"),
                )
            )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = (
            self.attention_dropout(
                attention_weights
            )
        )

        output = (
            attention_weights @ values
        )

        return output