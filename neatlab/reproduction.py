"""Offspring allocation and selection utilities for NEAT reproduction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .species import Species


@dataclass(frozen=True, slots=True)
class ReproductionConfig:
    """Configuration controlling offspring allocation and selection."""

    elitism: int
    survival_threshold: float

    def __post_init__(self) -> None:
        if self.elitism < 0:
            msg = "elitism must be >= 0."
            raise ValueError(msg)
        if not 0.0 < self.survival_threshold <= 1.0:
            msg = "survival_threshold must be in (0, 1]."
            raise ValueError(msg)


@dataclass(slots=True)
class ReproductionPlan:
    """Result of distributing offspring among species."""

    offspring: dict[int, int]
    elites: dict[int, tuple[int, ...]]
    selection_pool: dict[int, tuple[int, ...]]
    adjusted_fitness: dict[int, float]


def compute_offspring_allocation(
    species_list: Sequence[Species],
    fitnesses: Mapping[int, float],
    population_size: int,
    config: ReproductionConfig,
) -> ReproductionPlan:
    """Allocate offspring and determine elites for the next generation."""
    if population_size <= 0:
        msg = "population_size must be positive."
        raise ValueError(msg)
    if not species_list:
        msg = "At least one species is required."
        raise ValueError(msg)

    adjusted_fitness: dict[int, float] = {}
    species_totals: dict[int, float] = {}

    for species in species_list:
        if not species.members:
            msg = f"Species {species.id} has no members."
            raise ValueError(msg)
        size = len(species.members)
        total = 0.0
        for member_id in species.members:
            try:
                raw_fitness = fitnesses[member_id]
            except KeyError as error:
                msg = f"Missing fitness for genome id {error.args[0]}"
                raise KeyError(msg) from error
            adjusted = raw_fitness / size
            adjusted_fitness[member_id] = adjusted
            total += adjusted
        species_totals[species.id] = total

    total_adjusted = sum(species_totals.values())
    if total_adjusted <= 0.0:
        equal_share = population_size / len(species_totals)
        raw_allocations = dict.fromkeys(species_totals.keys(), equal_share)
    else:
        raw_allocations = {
            species_id: (species_totals[species_id] / total_adjusted) * population_size
            for species_id in species_totals
        }

    floors = {
        species_id: math.floor(value) for species_id, value in raw_allocations.items()
    }
    allocations = floors.copy()
    total_allocated = sum(allocations.values())
    remainder = population_size - total_allocated

    def fractional_part(species_id: int) -> float:
        return raw_allocations[species_id] - floors[species_id]

    ordering = sorted(floors, key=fractional_part, reverse=True)
    if remainder > 0:
        for species_id in ordering[:remainder]:
            allocations[species_id] += 1
    elif remainder < 0:
        for species_id in reversed(ordering):
            if remainder == 0:
                break
            available = allocations[species_id]
            if available <= 0:
                continue
            take = min(-remainder, available)
            allocations[species_id] -= take
            remainder += take

    total_allocated = sum(allocations.values())
    if total_allocated != population_size:
        # Fallback safeguard: adjust the first species to balance the total.
        first_species = ordering[0]
        allocations[first_species] += population_size - total_allocated

    elites: dict[int, tuple[int, ...]] = {}
    selection_pool: dict[int, tuple[int, ...]] = {}

    deficits: dict[int, int] = {}

    for species in species_list:
        members_sorted = sorted(
            species.members,
            key=lambda member_id: fitnesses[member_id],
            reverse=True,
        )
        elite_count = min(config.elitism, len(members_sorted))
        elites[species.id] = tuple(members_sorted[:elite_count])

        if allocations[species.id] < elite_count:
            deficits[species.id] = elite_count - allocations[species.id]
            allocations[species.id] = elite_count

        survivor_count = max(
            1,
            math.ceil(len(members_sorted) * config.survival_threshold),
        )
        selection_pool[species.id] = tuple(members_sorted[:survivor_count])

    total_deficit = sum(deficits.values())
    if total_deficit:
        surplus = {
            species.id: allocations[species.id] - len(elites[species.id])
            for species in species_list
        }
        available = sum(max(value, 0) for value in surplus.values())
        if total_deficit > available:
            msg = "Elitism requirements exceed population size."
            raise ValueError(msg)
        for species in sorted(
            species_list,
            key=lambda item: surplus[item.id],
            reverse=True,
        ):
            if total_deficit == 0:
                break
            capacity = surplus[species.id]
            if capacity <= 0:
                continue
            reduction = min(capacity, total_deficit)
            allocations[species.id] -= reduction
            surplus[species.id] -= reduction
            total_deficit -= reduction

    if total_deficit:
        msg = "Unable to satisfy elitism requirements after redistribution."
        raise RuntimeError(msg)

    if sum(allocations.values()) != population_size:
        msg = "Offspring allocation does not sum to the population size."
        raise RuntimeError(msg)

    return ReproductionPlan(
        offspring=allocations,
        elites=elites,
        selection_pool=selection_pool,
        adjusted_fitness=adjusted_fitness,
    )


__all__ = [
    "ReproductionConfig",
    "ReproductionPlan",
    "compute_offspring_allocation",
]
