from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(slots=True, frozen=True)
class BoundingBox:
    """
    Axis-aligned bounding box in image coordinates.

    Coordinates follow:

        x1,y1 ------ x2,y1
          |            |
          |            |
        x1,y2 ------ x2,y2
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1:
            raise ValueError(
                f"Invalid bounding box: x2 ({self.x2}) "
                f"must be greater than x1 ({self.x1})"
            )

        if self.y2 <= self.y1:
            raise ValueError(
                f"Invalid bounding box: y2 ({self.y2}) "
                f"must be greater than y1 ({self.y1})"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(slots=True, frozen=True)
class DetectedObject:
    """
    Single perception-engine output.
    """

    object_id: str
    label: str
    confidence: float
    bbox: BoundingBox
    attributes: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id cannot be empty")

        if not self.label:
            raise ValueError("label cannot be empty")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be within [0,1], got {self.confidence}"
            )


@dataclass(slots=True)
class ObjectSet:
    """
    Canonical output contract for the Perception Engine.

    This object is passed directly into the
    Scene Graph Engine.
    """

    schema_version: str
    image_id: str
    objects: List[DetectedObject]

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError("schema_version cannot be empty")

        if not self.image_id:
            raise ValueError("image_id cannot be empty")

        object_ids = {obj.object_id for obj in self.objects}

        if len(object_ids) != len(self.objects):
            raise ValueError(
                "Duplicate object IDs detected in ObjectSet"
            )

    @property
    def count(self) -> int:
        return len(self.objects)

    def get_by_id(self, object_id: str) -> DetectedObject:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj

        raise KeyError(
            f"Object '{object_id}' not found in ObjectSet"
        )

    def get_by_label(self, label: str) -> List[DetectedObject]:
        return [
            obj
            for obj in self.objects
            if obj.label == label
        ]