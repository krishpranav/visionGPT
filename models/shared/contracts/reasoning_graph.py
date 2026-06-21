from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True, frozen=True)
class ReasoningEvidence:
    """
    Reference to supporting evidence used
    during a reasoning step.

    Example:

        fact_001
        fact_002
    """

    reference_id: str

    def __post_init__(self) -> None:
        if not self.reference_id:
            raise ValueError(
                "reference_id cannot be empty"
            )


@dataclass(slots=True, frozen=True)
class ReasoningStep:
    """
    Single reasoning operation.

    Example:

        Step 1:
        Count all chairs.

        Step 2:
        Filter red chairs.

        Step 3:
        Return total.
    """

    step_number: int

    description: str

    evidence: List[ReasoningEvidence] = field(
        default_factory=list
    )

    confidence: float = 1.0

    def __post_init__(self) -> None:
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
                f"confidence must be within [0,1], got {self.confidence}"
            )


@dataclass(slots=True, frozen=True)
class Conclusion:
    """
    Final logical result produced by the
    Reasoning Engine.

    This is NOT natural language.

    Examples:

        7

        True

        obj_000012

        person_holding_umbrella
    """

    value: str

    confidence: float

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError(
                "value cannot be empty"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be within [0,1], got {self.confidence}"
            )


@dataclass(slots=True)
class ReasoningGraph:
    """
    Canonical output of the Reasoning Engine.

    Produced by:
        Reasoning Engine

    Consumed by:
        Response Engine
    """

    schema_version: str

    question: str

    steps: List[ReasoningStep]

    conclusion: Conclusion

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError(
                "schema_version cannot be empty"
            )

        if not self.question:
            raise ValueError(
                "question cannot be empty"
            )

        if not self.steps:
            raise ValueError(
                "ReasoningGraph requires at least one step"
            )

        expected = 1

        for step in self.steps:
            if step.step_number != expected:
                raise ValueError(
                    "Reasoning steps must be sequential "
                    f"(expected {expected}, got {step.step_number})"
                )

            expected += 1

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def confidence(self) -> float:
        """
        Aggregate confidence score for the graph.
        """

        if not self.steps:
            return self.conclusion.confidence

        step_confidence = (
            sum(
                step.confidence
                for step in self.steps
            )
            / len(self.steps)
        )

        return (
            step_confidence
            + self.conclusion.confidence
        ) / 2.0

    def get_step(
        self,
        step_number: int,
    ) -> ReasoningStep:
        for step in self.steps:
            if step.step_number == step_number:
                return step

        raise KeyError(
            f"Step {step_number} not found"
        )