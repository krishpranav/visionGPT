from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True, frozen=True)
class ReasoningEvidence:
    reference_id: str
    
    def __post_init__(self) -> None:
        if not self.reference_id:
            raise ValueError(
                "reference_id cannot be empty"
            )

@dataclass(slots=True, frozen=True)
class ReasoningStep:
    step_number: int
    description: str
    evidence: List[ReasoningEvidence] = field(
        default_factory=list
    )

    confidence: float = 1.0

    def __post_init_(self) -> None:
        if self.step_number < 1:
            raise ValueError(
                "step_number must be >= 1"
            )
        
        if not self.description:
            raise ValueError(
                "description cannot be empty"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be within [0,1] got {self.confidence}"
            )

