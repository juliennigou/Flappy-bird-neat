"""Population orchestration for NEAT evolutionary loop."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from random import Random

from .genome import Genome
from .reproduction import (
    ReproductionConfig,
    compute_offspring_allocation,
)
from .species import Species, SpeciesManager


@dataclass(frozen=True, slots=True)
class PopulationConfig:
    """Configuration values governing population evolution."""

    population_size: int
    elitism: int
    survival_threshold: float
    max_stagnation: int

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            msg = "population_size must be positive."
            raise ValueError(msg)
        if self.elitism < 0:
            msg = "elitism must be >= 0."
            raise ValueError(msg)
        if not 0.0 < self.survival_threshold <= 1.0:
            msg = "survival_threshold must be in (0, 1]."
            raise ValueError(msg)
        if self.max_stagnation < 0:
            msg = "max_stagnation must be >= 0."
            raise ValueError(msg)


@dataclass(slots=True)
class PopulationState:
    """Mutable state of the NEAT population."""

    generation: int
    genomes: dict[int, Genome]
    rng: Random
    species_manager: SpeciesManager
    config: PopulationConfig
    reproduction_config: ReproductionConfig
    fitnesses: dict[int, float] = field(default_factory=dict)
    champion_id: int | None = None
    champion_fitness: float = float("-inf")
    stagnant_species: set[int] = field(default_factory=set)

    def evaluate(
        self,
        evaluator: Callable[[Mapping[int, Genome], Random], Mapping[int, float]],
    ) -> Mapping[int, float]:
        """Evaluate all genomes and update fitness state."""

        results = evaluator(self.genomes, self.rng)
        if set(results) != set(self.genomes):
            msg = "Evaluator must return fitnesses for every genome."
            raise ValueError(msg)
        self.fitnesses = dict(results)

        best_id = max(self.fitnesses, key=self.fitnesses.__getitem__)
        best_fitness = self.fitnesses[best_id]
        if best_fitness > self.champion_fitness:
            self.champion_fitness = best_fitness
            self.champion_id = best_id
        return self.fitnesses

    def speciate(self) -> tuple[Species, ...]:
        """Assign genomes to species and handle stagnation accounting."""

        species = self.species_manager.speciate(
            self.genomes,
            generation=self.generation,
            fitnesses=self.fitnesses,
            rng=self.rng,
        )
        self.species_manager.adjust_threshold(self.generation, len(species))
        self._update_stagnation(species)
        return species

    def _update_stagnation(self, species: Sequence[Species]) -> None:
        self.stagnant_species.clear()
        for item in species:
            stagnation = self.generation - item.last_improved_generation
            if stagnation > self.config.max_stagnation:
                self.stagnant_species.add(item.id)

    def reproduce(
        self,
        species: Sequence[Species],
        mutate: Callable[[Genome, Random], None],
    ) -> None:
        """Generate the next generation of genomes."""

        eligible_species = [
            item for item in species if item.id not in self.stagnant_species
        ]
        if not eligible_species:
            eligible_species = list(species)

        plan = compute_offspring_allocation(
            eligible_species,
            self.fitnesses,
            population_size=self.config.population_size,
            config=self.reproduction_config,
        )

        new_genomes: dict[int, Genome] = {}

        # Carry over elites directly.
        for species_obj in eligible_species:
            elites = plan.elites.get(species_obj.id, ())
            for member_id in elites:
                new_genomes[member_id] = self.genomes[member_id].copy()

        # Fill the remaining offspring via mutation.
        for species_obj in eligible_species:
            quota = plan.offspring[species_obj.id]
            elites = plan.elites.get(species_obj.id, ())
            survivors = plan.selection_pool[species_obj.id]

            for _ in range(quota - len(elites)):
                parent_id = self.rng.choice(survivors)
                offspring = self.genomes[parent_id].copy()
                mutate(offspring, self.rng)
                new_id = self._allocate_genome_id(new_genomes)
                new_genomes[new_id] = offspring

        missing = self.config.population_size - len(new_genomes)
        if missing > 0:
            # Pad with random elites if insufficient offspring were produced.
            all_elites = [
                member
                for species_obj in eligible_species
                for member in plan.elites.get(species_obj.id, ())
            ]
            if not all_elites:
                all_elites = list(self.genomes)
            for _ in range(missing):
                clone_id = self.rng.choice(all_elites)
                new_id = self._allocate_genome_id(new_genomes)
                new_genomes[new_id] = self.genomes[clone_id].copy()

        self.genomes = new_genomes
        self.generation += 1

    def _allocate_genome_id(self, existing: Mapping[int, Genome]) -> int:
        candidate = 0
        occupied = set(existing) | set(self.genomes)
        while candidate in occupied:
            candidate += 1
        return candidate


__all__ = [
    "PopulationConfig",
    "PopulationState",
]
