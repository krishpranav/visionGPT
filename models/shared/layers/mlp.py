from __future__ import annotations

from torch import Tensor
from torch import nn

from models.shared.activations.swiglu import SwiGLU
from models.shared.layers.linear import Linear


class FeedForwardNetwork(nn.Module):
    """
    VisionGPT Feed Forward Network.

    Architecture

        Input
          │
          ▼
      Linear
          │
          ▼
       SwiGLU
          │
          ▼
      Dropout
          │
          ▼
      Linear
          │
          ▼
      Dropout
          │
          ▼
        Output

    Input
    -----

        [B, N, D]

    Output
    ------

        [B, N, D]
    """

    __constants__ = (
        "embedding_dim",
        "hidden_dimension",
    )

    def __init__(
        self,
        embedding_dim: int,
        hidden_dimension: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        if hidden_dimension <= 0:
            raise ValueError(
                "hidden_dimension must be greater than zero."
            )

        self.embedding_dim = embedding_dim
        self.hidden_dimension = hidden_dimension

        #
        # SwiGLU requires
        #
        # value | gate
        #
        self.input_projection = Linear(
            embedding_dim,
            hidden_dimension * 2,
        )

        self.activation = SwiGLU()

        self.output_projection = Linear(
            hidden_dimension,
            embedding_dim,
        )

        self.dropout = nn.Dropout(
            dropout,
        )

    def forward(
        self,
        inputs: Tensor,
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

        if inputs.ndim != 3:
            raise ValueError(
                "Expected tensor with shape "
                "[B, N, D]."
            )

        if inputs.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, "
                f"received {inputs.shape[-1]}."
            )

        outputs = self.input_projection(
            inputs,
        )

        outputs = self.activation(
            outputs,
        )

        outputs = self.dropout(
            outputs,
        )

        outputs = self.output_projection(
            outputs,
        )

        outputs = self.dropout(
            outputs,
        )

        return outputs

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, "
            f"hidden_dimension={self.hidden_dimension}"
        )