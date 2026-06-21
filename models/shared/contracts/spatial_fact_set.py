from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, List

class FactType(StrEnum):
    COUNT = "count"
    POSITION = "position"
    DISTANCE = "distance"
    GROUP = "group"
    LAYOUT = "layout"

@dataclass(slots=True, frozen=True)
class SpatialFact:
    fact_id: str
    fact_type: FactType
    statement: str
    value: Any
    confidence: float
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        if not self.fact_id:
            raise ValueError(
                "fact_id cannot be empty"
            )

        if not self.statement:
            raise ValueError(
                "statement cannot be empty"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be within [0,1] got {self.confidence}"
            )

@dataclass(slots=True)
class SpatialFactSet:
    schema_version: str
    facts: List[SpatialFact]

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError(
                "schema_version cannot be empty"
            )

        fact_ids = {
            fact.fact_id
            for fact in self.facts
        }

        if len(fact_ids) != len(self.facts):
            raise ValueError(
                "Duplicate fact IDs detected"
            )

    @property
    def count(self) -> int:
        return len(self.facts)

    def get_fact(
        self,
        fact_id: str
    ) -> SpatialFact:
        for fact in self.facts:
            if fact.fact_id == fact_id:
                return fact
            
        raise KeyError(
            f"Fact '{fact_id}' not found"
        )

    def facts_by_type(
        self,
        fact_type: FactType
    ) -> List[SpatialFact]:
        return [
            fact 
            for fact in self.facts
            if fact.fact_type == fact_type
        ]

    
