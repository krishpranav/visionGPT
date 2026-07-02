from __future__ import annotations

from torch import Tensor
from torch import nn

from models.perception.encoder import (
    TransformerEncoder,
)
from models.perception.tokenizer import (
    PatchTokenizer,
)
from models.shared.config.model_config import (
    ImageConfig,
    PerceptionConfig,
)


class PerceptionModel(nn.Module):
    """
    VisionGPT Perception Model.

    Pipeline

        Image
          │
          ▼
     Patch Tokenizer
          │
          ▼
    Transformer Encoder
          │
          ▼
    Encoded Patch Features
    """

    def __init__(
        self,
        image_config: ImageConfig,
        perception_config: PerceptionConfig,
    ) -> None:
        super().__init__()

        self.tokenizer = PatchTokenizer(
            image_channels=image_config.channels,
            patch_size=image_config.patch_size,
            embedding_dim=perception_config.embedding_dim,
        )

        self.encoder = TransformerEncoder(
            config=perception_config,
            num_patches=image_config.total_patches,
        )

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        images

            Shape:

                [B, C, H, W]

        Returns
        -------
        Tensor

            Shape:

                [B, N, D]
        """

        tokens = self.tokenizer(
            images,
        )

        outputs = self.encoder(
            tokens,
        )

        return outputs