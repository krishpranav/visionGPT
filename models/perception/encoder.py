from __future__ import annotations

from torch import Tensor
from torch import nn

from models.perception.encoder_block import (
    TransformerEncoderBlock,
)
from models.perception.position_embedding import (
    LearnablePositionEmbedding,
)
from models.shared.config.model_config import (
    PerceptionConfig,
)


class TransformerEncoder(nn.Module):
    """
    VisionGPT Perception Transformer Encoder.

    Pipeline

        Patch Tokens
             │
             ▼
    Position Embedding
             │
             ▼
      Transformer Block × N
             │
             ▼
       Encoded Patch Tokens
    """

    def __init__(
        self,
        config: PerceptionConfig,
        *,
        num_patches: int,
    ) -> None:
        super().__init__()

        if num_patches <= 0:
            raise ValueError(
                "num_patches must be greater than zero."
            )

        self.config = config
        self.num_patches = num_patches

        self.position_embedding = (
            LearnablePositionEmbedding(
                num_patches=num_patches,
                embedding_dim=config.embedding_dim,
            )
        )

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config=config,
                )
                for _ in range(
                    config.transformer_layers
                )
            ]
        )

    def forward(
        self,
        patch_embeddings: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        patch_embeddings

            Shape:
                [B, N, D]

        Returns
        -------
        Tensor

            Shape:
                [B, N, D]
        """

        if patch_embeddings.ndim != 3:
            raise ValueError(
                "Expected tensor with shape "
                "[B, N, D]."
            )

        outputs = self.position_embedding(
            patch_embeddings
        )

        for block in self.blocks:
            outputs = block(
                outputs,
                attention_mask=attention_mask,
            )

        return outputs

    def extra_repr(self) -> str:
        return (
            f"layers={self.config.transformer_layers}, "
            f"embedding_dim={self.config.embedding_dim}, "
            f"heads={self.config.attention_heads}"
        )