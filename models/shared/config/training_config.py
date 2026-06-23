from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """
    Optimizer configuration.
    """

    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """
    Learning-rate scheduling.
    """

    warmup_steps: int = 10_000
    total_steps: int = 1_000_000
    min_learning_rate: float = 3e-5
    schedule_type: str = "cosine"


@dataclass(frozen=True, slots=True)
class GradientConfig:
    """
    Gradient handling.
    """

    clip_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 8


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    """
    Numerical precision.
    """

    training_dtype: str = "bfloat16"
    master_weights_dtype: str = "float32"


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """
    Checkpoint strategy.
    """

    save_every_steps: int = 1000
    keep_hourly: int = 24
    keep_daily: int = 30
    keep_best: int = 10


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    """
    Distributed training configuration.
    """

    backend: str = "nccl"
    use_ddp: bool = True
    use_fsdp: bool = False
    use_activation_checkpointing: bool = True


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """
    Global training configuration.
    """

    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    gradients: GradientConfig = GradientConfig()
    precision: PrecisionConfig = PrecisionConfig()
    checkpoints: CheckpointConfig = CheckpointConfig()
    distributed: DistributedConfig = DistributedConfig()


DEFAULT_TRAINING_CONFIG = TrainingConfig()