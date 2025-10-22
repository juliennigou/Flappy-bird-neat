from __future__ import annotations

from random import Random

import pytest
from neatlab.genes import ConnectionGene, NodeGene, NodeType
from neatlab.genome import Genome
from neatlab.population import PopulationConfig, PopulationState
from neatlab.reproduction import ReproductionConfig
from neatlab.species import SpeciesConfig, SpeciesManager


def _make_genome(weight: float = 1.0) -> Genome:
    nodes = {
        0: NodeGene(0, NodeType.INPUT, "identity"),
        1: NodeGene(1, NodeType.OUTPUT, "identity"),
    }
    connections = {
        0: ConnectionGene(0, 0, 1, weight),
    }
    return Genome(nodes=nodes, connections=connections)


def _species_manager() -> SpeciesManager:
    return SpeciesManager(
        SpeciesConfig(
            c1=1.0,
            c2=1.0,
            c3=0.4,
            compatibility_threshold=3.0,
            target_species=2,
            species_adjust_period=1,
            adjust_rate=0.5,
            min_compatibility_threshold=0.5,
            max_compatibility_threshold=5.0,
        )
    )


def _population_state(pop_size: int = 4) -> PopulationState:
    genomes = {idx: _make_genome(weight=1.0 + idx * 0.1) for idx in range(pop_size)}
    return PopulationState(
        generation=0,
        genomes=genomes,
        rng=Random(0),
        species_manager=_species_manager(),
        config=PopulationConfig(
            population_size=pop_size,
            elitism=1,
            survival_threshold=0.5,
            max_stagnation=1,
        ),
        reproduction_config=ReproductionConfig(elitism=1, survival_threshold=0.5),
    )


def test_evaluate_updates_champion() -> None:
    population = _population_state()

    def evaluator(genomes, rng):  # type: ignore[override]
        return {gid: float(gid) for gid in genomes}

    population.evaluate(evaluator)
    assert population.champion_id == 3
    assert population.champion_fitness == pytest.approx(3.0)


def test_evaluate_requires_complete_results() -> None:
    population = _population_state(pop_size=2)

    def bad_evaluator(genomes, rng):  # type: ignore[override]
        return {next(iter(genomes)): 1.0}

    with pytest.raises(ValueError):
        population.evaluate(bad_evaluator)


def test_speciate_marks_stagnant_species_and_adjusts_threshold() -> None:
    population = _population_state(pop_size=3)
    population.fitnesses = dict.fromkeys(population.genomes, 1.0)

    species = population.speciate()
    assert len(species) == 1
    initial_threshold = population.species_manager.compatibility_threshold

    population.generation = 2
    population.speciate()
    assert population.stagnant_species == {species[0].id}
    assert population.species_manager.compatibility_threshold != initial_threshold


def test_reproduce_generates_next_generation_and_applies_mutation() -> None:
    population = _population_state()
    population.evaluate(lambda genomes, rng: {gid: 1.0 + gid for gid in genomes})
    species = population.speciate()

    mutated_weights: list[float] = []

    def mutate(genome: Genome, rng: Random) -> None:
        connection = next(iter(genome.connections.values()))
        mutated = connection.copy(weight=connection.weight + 0.5)
        genome.connections[mutated.innovation] = mutated
        mutated_weights.append(mutated.weight)

    population.reproduce(species, mutate)

    assert population.generation == 1
    assert len(population.genomes) == population.config.population_size
    assert len(set(population.genomes)) == population.config.population_size
    assert mutated_weights  # at least one mutation applied
    assert population.champion_id is not None


def test_reproduce_fallbacks_when_all_species_stagnant() -> None:
    population = _population_state(pop_size=2)
    population.evaluate(lambda genomes, rng: dict.fromkeys(genomes, 1.0))
    species = population.speciate()

    population.stagnant_species = {item.id for item in species}

    population.reproduce(species, lambda genome, rng: None)
    assert population.generation == 1
