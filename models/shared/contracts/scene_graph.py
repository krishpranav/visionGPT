from __future__ import annotations
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, List


class RelationType(StrEnum):
    """
    Supported relationship types for VisionGPT v1.
    """

    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"

    ABOVE = "above"
    BELOW = "below"

    HOLDING = "holding"
    TOUCHING = "touching"

    INSIDE = "inside"
    OUTSIDE = "outside"

    BEHIND = "behind"
    IN_FRONT_OF = "in_front_of"


@dataclass(slots=True, frozen=True)
class GraphNode:
    """
    Scene graph entity node.
    """

    object_id: str
    label: str
    attributes: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id cannot be empty")

        if not self.label:
            raise ValueError("label cannot be empty")


@dataclass(slots=True, frozen=True)
class GraphEdge:
    """
    Directed relationship between two nodes.
    """

    source_id: str
    target_id: str
    relation: RelationType
    confidence: float

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id cannot be empty")

        if not self.target_id:
            raise ValueError("target_id cannot be empty")

        if self.source_id == self.target_id:
            raise ValueError(
                "source_id and target_id cannot be identical"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be within [0,1], got {self.confidence}"
            )


@dataclass(slots=True)
class SceneGraph:
    """
    Canonical Scene Graph representation.

    Produced by:
        Scene Graph Engine

    Consumed by:
        Spatial Engine
        Reasoning Engine
    """

    schema_version: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError(
                "schema_version cannot be empty"
            )

        node_ids = {node.object_id for node in self.nodes}

        if len(node_ids) != len(self.nodes):
            raise ValueError(
                "Duplicate node IDs detected"
            )

        for edge in self.edges:
            if edge.source_id not in node_ids:
                raise ValueError(
                    f"Unknown source node '{edge.source_id}'"
                )

            if edge.target_id not in node_ids:
                raise ValueError(
                    f"Unknown target node '{edge.target_id}'"
                )

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def get_node(self, object_id: str) -> GraphNode:
        for node in self.nodes:
            if node.object_id == object_id:
                return node

        raise KeyError(
            f"Node '{object_id}' not found"
        )

    def outgoing_edges(
        self,
        object_id: str,
    ) -> List[GraphEdge]:
        return [
            edge
            for edge in self.edges
            if edge.source_id == object_id
        ]

    def incoming_edges(
        self,
        object_id: str,
    ) -> List[GraphEdge]:
        return [
            edge
            for edge in self.edges
            if edge.target_id == object_id
        ]

    def edges_by_relation(
        self,
        relation: RelationType,
    ) -> List[GraphEdge]:
        return [
            edge
            for edge in self.edges
            if edge.relation == relation
        ]