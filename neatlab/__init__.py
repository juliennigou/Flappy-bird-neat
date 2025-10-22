"""Core NEAT primitives for reusable neuroevolution workflows."""

from __future__ import annotations

from .benchmark import BenchmarkResult, run_benchmark
from .evaluator import (
    EvaluationConfig,
    EvaluationStats,
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
from .metrics import MetricsRow, MetricsWriter
from .network import (
    DEFAULT_ACTIVATIONS,
    FeedForwardNetwork,
    compute_feedforward_layers,
)
from .population import (
    MutationOperators,
    PopulationConfig,
    PopulationState,
)
from .reproduction import (
    ReproductionConfig,
    ReproductionPlan,
    compute_offspring_allocation,
)
from .persistence import TrainingCheckpoint, load_checkpoint, save_checkpoint
from .reporters import EventLogger
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
    "BenchmarkResult",
    "run_benchmark",
    "EvaluationConfig",
    "EvaluationStats",
    "SyncEvaluator",
    "ParallelEvaluator",
    "MetricsRow",
    "MetricsWriter",
    "Species",
    "SpeciesConfig",
    "SpeciesManager",
    "compatibility_distance",
    "ReproductionConfig",
    "ReproductionPlan",
    "compute_offspring_allocation",
    "PopulationConfig",
    "PopulationState",
    "MutationOperators",
    "TrainingCheckpoint",
    "save_checkpoint",
    "load_checkpoint",
    "EventLogger",
]
