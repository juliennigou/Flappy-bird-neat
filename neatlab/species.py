"""Species management and compatibility utilities for NEAT."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from random import Random

from .genome import Genome


@dataclass(frozen=True, slots=True)
class SpeciesConfig:
    """Configuration parameters controlling speciation behaviour."""

    c1: float
    c2: float
    c3: float
    compatibility_threshold: float
    target_species: int
    species_adjust_period: int
    adjust_rate: float = 0.1
    min_compatibility_threshold: float = 0.1
    max_compatibility_threshold: float = 10.0

    def __post_init__(self) -> None:
        if self.c1 < 0 or self.c2 < 0 or self.c3 < 0:
            msg = "Compatibility coefficients must be non-negative."
            raise ValueError(msg)
        if self.compatibility_threshold <= 0:
            msg = "compatibility_threshold must be positive."
            raise ValueError(msg)
        if self.target_species <= 0:
            msg = "target_species must be positive."
            raise ValueError(msg)
        if self.species_adjust_period < 0:
            msg = "species_adjust_period must be >= 0."
            raise ValueError(msg)
        if self.adjust_rate <= 0:
            msg = "adjust_rate must be positive."
            raise ValueError(msg)
        if self.min_compatibility_threshold <= 0:
            msg = "min_compatibility_threshold must be positive."
            raise ValueError(msg)
        if self.max_compatibility_threshold <= 0:
            msg = "max_compatibility_threshold must be positive."
            raise ValueError(msg)
        if self.min_compatibility_threshold >= self.max_compatibility_threshold:
            msg = "min_compatibility_threshold must be < max_compatibility_threshold."
            raise ValueError(msg)


def compatibility_distance(
    left: Genome,
    right: Genome,
    *,
    c1: float,
    c2: float,
    c3: float,
) -> float:
    """Compute the NEAT compatibility distance between two genomes."""
    innovations_left = sorted(left.connections)
    innovations_right = sorted(right.connections)

    index_left = 0
    index_right = 0
    disjoint = 0
    excess = 0
    weight_diff_sum = 0.0
    matches = 0

    while index_left < len(innovations_left) and index_right < len(innovations_right):
        innov_left = innovations_left[index_left]
        innov_right = innovations_right[index_right]
        if innov_left == innov_right:
            matches += 1
            weight_diff_sum += abs(
                left.connections[innov_left].weight
                - right.connections[innov_right].weight
            )
            index_left += 1
            index_right += 1
        elif innov_left < innov_right:
            disjoint += 1
            index_left += 1
        else:
            disjoint += 1
            index_right += 1

    excess += len(innovations_left) - index_left
    excess += len(innovations_right) - index_right

    n = max(len(innovations_left), len(innovations_right))
    n = 1 if n < 20 else n
    average_weight_diff = weight_diff_sum / matches if matches else 0.0

    return (c1 * excess / n) + (c2 * disjoint / n) + (c3 * average_weight_diff)


@dataclass(slots=True)
class Species:
    """Tracks a group of genetically similar genomes."""

    id: int
    representative: Genome
    creation_generation: int
    members: list[int] = field(default_factory=list)
    best_fitness: float = float("-inf")
    last_improved_generation: int = field(init=False)
    age: int = field(init=False)

    def __post_init__(self) -> None:
        self.last_improved_generation = self.creation_generation
        self.age = 0

    def clear_members(self) -> None:
        """Remove all member identifiers."""
        self.members.clear()

    def set_members(self, members: Sequence[int]) -> None:
        """Assign members from an iterable of genome ids."""
        self.members = list(members)


@dataclass(slots=True)
class SpeciesManager:
    """Maintains species representatives and membership assignments."""

    config: SpeciesConfig
    compatibility_threshold: float = field(init=False)
    _species: dict[int, Species] = field(init=False, default_factory=dict)
    _next_species_id: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.compatibility_threshold = self.config.compatibility_threshold

    def distance(self, left: Genome, right: Genome) -> float:
        """Compute compatibility distance using the configured coefficients."""
        return compatibility_distance(
            left,
            right,
            c1=self.config.c1,
            c2=self.config.c2,
            c3=self.config.c3,
        )

    def adjust_threshold(self, generation: int, species_count: int) -> None:
        """Auto-adjust the compatibility threshold towards the target species count."""
        if self.config.species_adjust_period == 0:
            return
        if generation % self.config.species_adjust_period != 0:
            return
        if species_count == 0:
            return

        if species_count < self.config.target_species:
            self.compatibility_threshold = max(
                self.config.min_compatibility_threshold,
                self.compatibility_threshold - self.config.adjust_rate,
            )
        elif species_count > self.config.target_species:
            self.compatibility_threshold = min(
                self.config.max_compatibility_threshold,
                self.compatibility_threshold + self.config.adjust_rate,
            )

    def speciate(
        self,
        genomes: Mapping[int, Genome],
        generation: int,
        *,
        fitnesses: Mapping[int, float] | None = None,
        rng: Random,
    ) -> tuple[Species, ...]:
        """Assign genomes to species based on compatibility distance."""
        if not genomes:
            self._species.clear()
            return ()

        for species in self._species.values():
            species.clear_members()

        for genome_id in sorted(genomes):
            genome = genomes[genome_id]
            matched = self._find_species(genome)
            if matched is None:
                matched = self._create_species(genome, generation)
                self._species[matched.id] = matched
            matched.members.append(genome_id)

        empty_species = [
            species_id
            for species_id, species in self._species.items()
            if not species.members
        ]
        for species_id in empty_species:
            del self._species[species_id]

        for species in self._species.values():
            species.age = generation - species.creation_generation
            representative_id = rng.choice(species.members)
            species.representative = genomes[representative_id].copy()
            species.members.sort()

            if fitnesses is None:
                continue
            try:
                best_member = max(
                    species.members,
                    key=lambda member_id: fitnesses[member_id],
                )
            except KeyError as error:
                msg = f"Missing fitness for genome id {error.args[0]}"
                raise KeyError(msg) from error
            best_fitness = fitnesses[best_member]
            if best_fitness > species.best_fitness:
                species.best_fitness = best_fitness
                species.last_improved_generation = generation

        return tuple(sorted(self._species.values(), key=lambda item: item.id))

    def species(self) -> tuple[Species, ...]:
        """Return the tracked species sorted by identifier."""
        return tuple(sorted(self._species.values(), key=lambda item: item.id))

    def _find_species(self, genome: Genome) -> Species | None:
        for species in self._species.values():
            distance = self.distance(genome, species.representative)
            if distance <= self.compatibility_threshold:
                return species
        return None

    def _create_species(self, genome: Genome, generation: int) -> Species:
        species_id = self._next_species_id
        self._next_species_id += 1
        return Species(
            id=species_id,
            representative=genome.copy(),
            creation_generation=generation,
        )


__all__ = [
    "Species",
    "SpeciesConfig",
    "SpeciesManager",
    "compatibility_distance",
]
