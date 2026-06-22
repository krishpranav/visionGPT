from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImageConfig:
    channels: int = 3
    resolution: int = 1024
    patch_size: int = 16

    @property
    def patches_per_side(self) -> int:
        return self.resolution // self.patch_size

    @property
    def total_patches(self) -> int:
        side = self.patches_per_side
        return side * side


@dataclass(frozen=True, slots=True)
class PerceptionConfig:
    """
    Vision backbone configuration.

    These values define architecture only.
    They do not imply trained weights.
    """

    embedding_dim: int = 2048
    transformer_layers: int = 32
    attention_heads: int = 32
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0


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
    Reasoning engine configuration.
    """

    embedding_dim: int = 3072
    transformer_layers: int = 40
    attention_heads: int = 48
    mlp_ratio: float = 4.0
    max_reasoning_steps: int = 32
    max_evidence_per_step: int = 64
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class ResponseConfig:
    """
    Response generation limits.
    """

    max_response_tokens: int = 512
    max_answer_length: int = 2048


@dataclass(frozen=True, slots=True)
class VisionGPTConfig:
    """
    Master configuration object.

    This becomes the single source of truth
    for every VisionGPT component.
    """

    image: ImageConfig = ImageConfig()
    perception: PerceptionConfig = PerceptionConfig()
    scene_graph: SceneGraphConfig = SceneGraphConfig()
    spatial: SpatialConfig = SpatialConfig()
    reasoning: ReasoningConfig = ReasoningConfig()
    response: ResponseConfig = ResponseConfig()


DEFAULT_CONFIG = VisionGPTConfig()