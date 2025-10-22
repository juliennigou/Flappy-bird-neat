"""Core NEAT primitives for reusable neuroevolution workflows."""

from __future__ import annotations

from .evaluator import (
    EvaluationConfig,
    ParallelEvaluator,
    SyncEvaluator,
)
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
from .population import (
    PopulationConfig,
    PopulationState,
)
from .reproduction import (
    ReproductionConfig,
    ReproductionPlan,
    compute_offspring_allocation,
)
from .species import (
    Species,
    SpeciesConfig,
    SpeciesManager,
    compatibility_distance,
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
    "EvaluationConfig",
    "SyncEvaluator",
    "ParallelEvaluator",
    "Species",
    "SpeciesConfig",
    "SpeciesManager",
    "compatibility_distance",
    "ReproductionConfig",
    "ReproductionPlan",
    "compute_offspring_allocation",
    "PopulationConfig",
    "PopulationState",
]
