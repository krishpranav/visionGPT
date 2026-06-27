from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum


class ActivationType(StrEnum):
    """
    Supported activation functions.
    """

    SWIGLU = "swiglu"
    GELU = "gelu"
    RELU = "relu"


class NormalizationType(StrEnum):
    """
    Supported normalization layers.
    """

    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"


class AttentionType(StrEnum):
    """
    Supported attention implementations.
    """

    GLOBAL = "global"
    WINDOW = "window"
    FLASH = "flash"


@dataclass(frozen=True, slots=True)
class ImageConfig:
    """
    Image processing configuration.
    """

    channels: int = 3
    resolution: int = 1024
    patch_size: int = 16

    def __post_init__(self) -> None:
        if self.resolution % self.patch_size != 0:
            raise ValueError(
                "resolution must be divisible by patch_size"
            )

    @property
    def patches_per_side(self) -> int:
        return self.resolution // self.patch_size

    @property
    def total_patches(self) -> int:
        return self.patches_per_side ** 2

@dataclass(frozen=True, slots=True)
class PerceptionConfig:
    """
    Vision backbone configuration.
    """

    embedding_dim: int = 2048
    transformer_layers: int = 32
    attention_heads: int = 32
    hidden_multiplier: float = 8.0 / 3.0
    hidden_multiple_of: int = 256
    activation: ActivationType = ActivationType.SWIGLU

    normalization: NormalizationType = (
        NormalizationType.RMS_NORM
    )

    attention: AttentionType = (
        AttentionType.GLOBAL
    )

    dropout: float = 0.0
    attention_dropout: float = 0.0

    @property
    def hidden_dimension(self) -> int:
        raw = int(
            self.embedding_dim
            * self.hidden_multiplier
        )

        remainder = (
            raw % self.hidden_multiple_of
        )

        if remainder == 0:
            return raw

        return (
            raw
            + (
                self.hidden_multiple_of
                - remainder
            )
        )

@dataclass(frozen=True, slots=True)
class SceneGraphConfig:
    """
    Scene graph generation configuration.
    """

    max_objects: int = 512
    max_relationships: int = 4096
    relation_embedding_dim: int = 512


@dataclass(frozen=True, slots=True)
class SpatialConfig:
    """
    Spatial reasoning limits.
    """

    max_facts: int = 2048
    max_countable_objects: int = 1000
    distance_precision: int = 2

@dataclass(frozen=True, slots=True)
class ReasoningConfig:
    """
    Reasoning transformer configuration.
    """

    embedding_dim: int = 3072
    transformer_layers: int = 40
    attention_heads: int = 48
    hidden_multiplier: float = 8.0 / 3.0
    hidden_multiple_of: int = 256
    activation: ActivationType = ActivationType.SWIGLU

    normalization: NormalizationType = (
        NormalizationType.RMS_NORM
    )

    attention: AttentionType = (
        AttentionType.GLOBAL
    )

    max_reasoning_steps: int = 32
    max_evidence_per_step: int = 64
    dropout: float = 0.0
    attention_dropout: float = 0.0

    @property
    def hidden_dimension(self) -> int:
        raw = int(
            self.embedding_dim
            * self.hidden_multiplier
        )

        remainder = (
            raw % self.hidden_multiple_of
        )

        if remainder == 0:
            return raw

        return (
            raw
            + (
                self.hidden_multiple_of
                - remainder
            )
        )

@dataclass(frozen=True, slots=True)
class ResponseConfig:
    """
    Response generation configuration.
    """

    max_response_tokens: int = 512
    max_answer_length: int = 2048

@dataclass(frozen=True, slots=True)
class VisionGPTConfig:
    """
    Master configuration object.
    """

    image: ImageConfig = ImageConfig()

    perception: PerceptionConfig = (
        PerceptionConfig()
    )

    scene_graph: SceneGraphConfig = (
        SceneGraphConfig()
    )

    spatial: SpatialConfig = (
        SpatialConfig()
    )

    reasoning: ReasoningConfig = (
        ReasoningConfig()
    )

    response: ResponseConfig = (
        ResponseConfig()
    )


DEFAULT_CONFIG = VisionGPTConfig()