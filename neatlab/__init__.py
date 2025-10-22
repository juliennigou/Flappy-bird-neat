"""Core NEAT primitives for reusable neuroevolution workflows."""

from __future__ import annotations

from .genes import ConnectionGene, NodeGene, NodeType
from .genome import (
    AddConnectionConfig,
    AddNodeConfig,
    CrossoverConfig,
    Genome,
    WeightMutationConfig,
)
from .innovations import InnovationSnapshot, InnovationTracker
from .network import (
    DEFAULT_ACTIVATIONS,
    FeedForwardNetwork,
    compute_feedforward_layers,
)

__all__ = [
    "ConnectionGene",
    "InnovationSnapshot",
    "InnovationTracker",
    "NodeGene",
    "NodeType",
    "Genome",
    "WeightMutationConfig",
    "AddConnectionConfig",
    "AddNodeConfig",
    "CrossoverConfig",
    "FeedForwardNetwork",
    "DEFAULT_ACTIVATIONS",
    "compute_feedforward_layers",
]
