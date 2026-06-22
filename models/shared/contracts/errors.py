from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ErrorContext:
    component: str
    operation: str
    details: dict[str, Any]


class VisionGPTError(Exception):
    """
    Base exception for all VisionGPT failures.
    """

    def __init__(
        self,
        message: str,
        context: ErrorContext | None = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.context = context

    def __str__(self) -> str:
        if self.context is None:
            return self.message

        return (
            f"{self.message} "
            f"[component={self.context.component}, "
            f"operation={self.context.operation}]"
        )


class ValidationError(VisionGPTError):
    """
    Invalid schema or contract data.
    """


class ConfigurationError(VisionGPTError):
    """
    Invalid runtime or model configuration.
    """


class DatasetError(VisionGPTError):
    """
    Dataset corruption or dataset loading failure.
    """


class ModelInitializationError(VisionGPTError):
    """
    Model construction failure.
    """


class CheckpointError(VisionGPTError):
    """
    Checkpoint loading or saving failure.
    """


class InferenceError(VisionGPTError):
    """
    Runtime inference failure.
    """


class PerceptionError(VisionGPTError):
    """
    Perception engine failure.
    """


class SceneGraphError(VisionGPTError):
    """
    Scene graph generation failure.
    """


class SpatialReasoningError(VisionGPTError):
    """
    Spatial reasoning failure.
    """


class ReasoningError(VisionGPTError):
    """
    Logical reasoning failure.
    """


class ResponseGenerationError(VisionGPTError):
    """
    Response engine failure.
    """