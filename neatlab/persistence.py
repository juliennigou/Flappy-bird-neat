"""Checkpoint helpers for saving and resuming training sessions."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .genome import Genome
from .innovations import InnovationSnapshot
from .population import PopulationState


@dataclass(slots=True)
class TrainingCheckpoint:
    """Serializable representation of a training session."""

    generation: int
    population_state: PopulationState
    best_genome: Genome
    best_fitness: float
    innovation_snapshot: InnovationSnapshot = field(
        default_factory=lambda: InnovationSnapshot(next_innovation=0, pairs=())
    )


def save_checkpoint(path: Path, checkpoint: TrainingCheckpoint) -> None:
    """Persist a training checkpoint to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: Path) -> TrainingCheckpoint:
    """Load a previously saved training checkpoint."""
    source = Path(path)
    with source.open("rb") as handle:
        data: Any = pickle.load(handle)
    if not isinstance(data, TrainingCheckpoint):
        msg = f"Invalid checkpoint payload in {source}"
        raise ValueError(msg)
    if not hasattr(data, "innovation_snapshot"):
        object.__setattr__(
            data,
            "innovation_snapshot",
            InnovationSnapshot(next_innovation=0, pairs=()),
        )
    return data


__all__ = ["TrainingCheckpoint", "save_checkpoint", "load_checkpoint"]
