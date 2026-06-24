from __future__ import annotations
import torch
from torch import Tensor
from torch import nn

class LearnablePositionEmbedding(nn.Module):
    def __init__(
        self,
        num_patches: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        if num_patches <= 0:
            raise ValueError(
                "num_patches must be > 0"
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be > 0"
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
        nn.init.trunc_normal_(
            self.position_embeddings,
            mean = 0.0,
            std = 0.02,
        )

    def forward(
        self,
        tokens: Tensor
    ) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                "Expected token with shape"
                "[B, N, D]"
            )

        _, num_tokens, embedding_dim = (
            tokens.shape
        )

        if num_tokens != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches}"
                f"tokens but received"
                f"{num_tokens}"
            )

    
    @property
    def shape(
        self
    ) -> tuple[int, int]:
        return (
            self.num_patches,
            self.embedding_dim,
        )