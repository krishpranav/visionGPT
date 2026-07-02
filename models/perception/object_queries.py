from __future__ import annotations
import torch
from torch import Tensor
from torch import nn
from models.shared.initialization.weight_init import (
    initialize_embedding,
)


class ObjectQueries(nn.Module):
    """
    Learnable object query embeddings.

    These queries are consumed by the object
    decoder to extract object-centric features
    from encoded patch representations.
    """

    __constants__ = (
        "num_queries",
        "embedding_dim",
    )

    def __init__(
        self,
        num_queries: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        if num_queries <= 0:
            raise ValueError(
                "num_queries must be greater than zero."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        self.num_queries = num_queries
        self.embedding_dim = embedding_dim

        self.queries = nn.Parameter(
            torch.empty(
                num_queries,
                embedding_dim,
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_embedding(
            self.queries,
        )

    def forward(
        self,
        batch_size: int,
    ) -> Tensor:
        """
        Returns
        -------
        Tensor

            Shape:

                [B, Q, D]
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        return (
            self.queries
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

    def extra_repr(self) -> str:
        return (
            f"num_queries={self.num_queries}, "
            f"embedding_dim={self.embedding_dim}"
        )