"""Gene primitives (nodes and connections) for NEAT genomes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class NodeType(str, Enum):
    """Enumeration of supported node categories."""

    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"
    BIAS = "bias"

    @classmethod
    def coerce(cls, value: NodeType | str) -> NodeType:
        """Coerce a string or NodeType into a NodeType instance."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            msg = f"Unsupported node type value: {value!r}"
            raise TypeError(msg)
        try:
            return cls(value.lower())
        except ValueError as error:
            valid = ", ".join(member.value for member in cls)
            msg = f"Invalid node type {value!r}. Expected one of: {valid}"
            raise ValueError(msg) from error


@dataclass(frozen=True, slots=True)
class NodeGene:
    """Represents a node within a NEAT genome."""

    id: int
    type: NodeType
    activation: str

    def __post_init__(self) -> None:
        if self.id < 0:
            msg = "Node id must be non-negative."
            raise ValueError(msg)
        coerced_type = NodeType.coerce(self.type)
        object.__setattr__(self, "type", coerced_type)
        if not isinstance(self.activation, str) or not self.activation.strip():
            msg = "Activation must be a non-empty string."
            raise ValueError(msg)
        object.__setattr__(self, "activation", self.activation.strip())

    def copy(
        self,
        *,
        node_id: int | None = None,
        activation: str | None = None,
    ) -> NodeGene:
        """Return a copy of the node with optional overrides."""
        return NodeGene(
            id=self.id if node_id is None else node_id,
            type=self.type,
            activation=self.activation if activation is None else activation,
        )


@dataclass(frozen=True, slots=True)
class ConnectionGene:
    """Represents a connection between two nodes in a NEAT genome."""

    innovation: int
    in_node_id: int
    out_node_id: int
    weight: float
    enabled: bool = True

    def __post_init__(self) -> None:
        for field_name, value in (
            ("innovation", self.innovation),
            ("in_node_id", self.in_node_id),
            ("out_node_id", self.out_node_id),
        ):
            if value < 0:
                msg = f"{field_name} must be non-negative."
                raise ValueError(msg)
        try:
            weight = float(self.weight)
        except (TypeError, ValueError) as error:
            msg = f"weight must be convertible to float, got {self.weight!r}"
            raise ValueError(msg) from error
        if not math.isfinite(weight):
            msg = "weight must be a finite number."
            raise ValueError(msg)
        object.__setattr__(self, "weight", weight)

    def copy(
        self,
        *,
        weight: float | None = None,
        enabled: bool | None = None,
        in_node_id: int | None = None,
        out_node_id: int | None = None,
    ) -> ConnectionGene:
        """Return a copy with optional field overrides."""
        return ConnectionGene(
            innovation=self.innovation,
            in_node_id=self.in_node_id if in_node_id is None else in_node_id,
            out_node_id=self.out_node_id if out_node_id is None else out_node_id,
            weight=self.weight if weight is None else float(weight),
            enabled=self.enabled if enabled is None else enabled,
        )

    def toggled(self) -> ConnectionGene:
        """Return a copy with the `enabled` flag flipped."""
        return self.copy(enabled=not self.enabled)


__all__ = ["NodeType", "NodeGene", "ConnectionGene"]
