from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

from models.shared.initialization.weight_init import (
    initialize_embedding,
)


class LearnablePositionEmbedding(nn.Module):
    """
    Learnable absolute position embedding.

    This module injects positional information into
    patch embeddings produced by the PatchTokenizer.

    Input
    -----
        Tensor
            Shape:
                [B, N, D]

    Output
    ------
        Tensor
            Shape:
                [B, N, D]
    """

    __constants__ = (
        "num_patches",
        "embedding_dim",
    )

    def __init__(
        self,
        num_patches: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        if num_patches <= 0:
            raise ValueError(
                "num_patches must be greater than zero."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        self.num_patches = num_patches
        self.embedding_dim = embedding_dim

        self.position_embeddings = nn.Parameter(
            torch.empty(
                1,
                num_patches,
                embedding_dim,
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize learnable position embeddings.

        Initialization is delegated to the centralized
        VisionGPT initialization subsystem.
        """

        initialize_embedding(
            self.position_embeddings
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
                "Expected tensor with shape "
                "[B, N, D]."
            )

        _, num_tokens, embedding_dim = tokens.shape

        if num_tokens != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} tokens, "
                f"received {num_tokens}."
            )

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, "
                f"received {embedding_dim}."
            )

        return tokens + self.position_embeddings.to(
            dtype=tokens.dtype,
            device=tokens.device,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """
        Returns
        -------
        tuple[int, int]

            (num_patches, embedding_dim)
        """

        return (
            self.num_patches,
            self.embedding_dim,
        )

    def extra_repr(self) -> str:
        return (
            f"num_patches={self.num_patches}, "
            f"embedding_dim={self.embedding_dim}"
        )