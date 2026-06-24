from __future__ import annotations

import torch
from torch import Tensor
from torch import nn


class PatchTokenizer(nn.Module):
    """
    VisionGPT patch tokenizer.

    Converts an image tensor into a sequence
    of patch embeddings.

    Input:

        [B, C, H, W]

    Output:

        [B, N, D]

    Where:

        B = batch size

        N = number of patches

        D = embedding dimension
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                "image_size must be divisible "
                "by patch_size"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.grid_size = image_size // patch_size

        self.num_patches = (
            self.grid_size * self.grid_size
        )

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
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

        if images.ndim != 4:
            raise ValueError(
                "Expected image tensor with "
                "shape [B, C, H, W]"
            )

        batch_size, channels, height, width = (
            images.shape
        )

        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} "
                f"channels but received "
                f"{channels}"
            )

        if height != self.image_size:
            raise ValueError(
                f"Expected image height "
                f"{self.image_size} but "
                f"received {height}"
            )

        if width != self.image_size:
            raise ValueError(
                f"Expected image width "
                f"{self.image_size} but "
                f"received {width}"
            )

        tokens = self.projection(images)
        tokens = tokens.flatten(2)
        tokens = tokens.transpose(1, 2)

        return tokens

    @property
    def output_shape(
        self,
    ) -> tuple[int, int]:
        """
        Returns:

            (num_patches, embedding_dim)
        """

        return (
            self.num_patches,
            self.embedding_dim,
        )