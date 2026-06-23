from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """
    Runtime execution device.
    """

    device: str = "cuda"
    allow_cpu_fallback: bool = True


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """
    Inference behavior.
    """

    batch_size: int = 1
    max_concurrent_requests: int = 64
    confidence_threshold: float = 0.60


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """
    Runtime memory limits.
    """

    max_vram_gb: int = 24
    enable_attention_cache: bool = True
    enable_graph_cache: bool = True


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """
    Runtime observability.
    """

    enable_metrics: bool = True
    enable_tracing: bool = True
    log_level: str = "INFO"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """
    Global runtime configuration.
    """

    device: DeviceConfig = DeviceConfig()
    inference: InferenceConfig = InferenceConfig()
    memory: MemoryConfig = MemoryConfig()
    logging: LoggingConfig = LoggingConfig()


DEFAULT_RUNTIME_CONFIG = RuntimeConfig()