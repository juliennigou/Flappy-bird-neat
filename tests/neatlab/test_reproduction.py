from __future__ import annotations

import pytest
from neatlab.genes import NodeGene, NodeType
from neatlab.genome import Genome
from neatlab.reproduction import (
    ReproductionConfig,
    compute_offspring_allocation,
)
from neatlab.species import Species


def example_genome() -> Genome:
    nodes = {
        0: NodeGene(0, NodeType.INPUT, "identity"),
        1: NodeGene(1, NodeType.OUTPUT, "identity"),
    }
    return Genome(nodes=nodes, connections={})


def make_species(species_id: int, members: list[int]) -> Species:
    species = Species(
        id=species_id,
        representative=example_genome(),
        creation_generation=0,
    )
    species.set_members(members)
    return species


def test_offspring_allocation_respects_population_and_elitism() -> None:
    species_a = make_species(0, [1, 2, 3])
    species_b = make_species(1, [4, 5])
    fitnesses = {1: 1.0, 2: 1.5, 3: 0.5, 4: 3.0, 5: 2.5}
    config = ReproductionConfig(elitism=1, survival_threshold=0.5)

    plan = compute_offspring_allocation(
        [species_a, species_b],
        fitnesses,
        population_size=6,
        config=config,
    )

    assert sum(plan.offspring.values()) == 6
    assert plan.elites[0] == (2,)
    assert plan.elites[1] == (4,)
    assert plan.selection_pool[0] == (2, 1)
    assert plan.selection_pool[1] == (4,)


def test_offspring_allocation_zero_adjusted_fitness_splits_evenly() -> None:
    species_a = make_species(0, [1, 2])
    species_b = make_species(1, [3, 4])
    fitnesses = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    config = ReproductionConfig(elitism=0, survival_threshold=0.5)

    plan = compute_offspring_allocation(
        [species_a, species_b],
        fitnesses,
        population_size=4,
        config=config,
    )

    assert plan.offspring[0] == 2
    assert plan.offspring[1] == 2


def test_offspring_allocation_raises_when_elitism_exceeds_population() -> None:
    species_a = make_species(0, [1, 2])
    species_b = make_species(1, [3, 4])
    fitnesses = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    config = ReproductionConfig(elitism=2, survival_threshold=1.0)

    with pytest.raises(ValueError):
        compute_offspring_allocation(
            [species_a, species_b],
            fitnesses,
            population_size=3,
            config=config,
        )


def test_reproduction_config_validation_errors() -> None:
    with pytest.raises(ValueError):
        ReproductionConfig(elitism=-1, survival_threshold=0.5)
    with pytest.raises(ValueError):
        ReproductionConfig(elitism=0, survival_threshold=0.0)


def test_offspring_allocation_requires_valid_inputs() -> None:
    config = ReproductionConfig(elitism=0, survival_threshold=0.5)

    with pytest.raises(ValueError):
        compute_offspring_allocation([], {}, population_size=2, config=config)

    empty_species = make_species(0, [])
    with pytest.raises(ValueError):
        compute_offspring_allocation(
            [empty_species],
            {},
            population_size=2,
            config=config,
        )

    species_a = make_species(0, [1])
    with pytest.raises(KeyError):
        compute_offspring_allocation([species_a], {}, population_size=2, config=config)

    with pytest.raises(ValueError):
        compute_offspring_allocation(
            [species_a],
            {1: 1.0},
            population_size=0,
            config=config,
        )


def test_offspring_allocation_rebalances_for_elite_deficit() -> None:
    species_a = make_species(0, [1, 2])
    species_b = make_species(1, [3])
    fitnesses = {1: 1.0, 2: 0.5, 3: 10.0}
    config = ReproductionConfig(elitism=2, survival_threshold=0.5)

    plan = compute_offspring_allocation(
        [species_a, species_b],
        fitnesses,
        population_size=3,
        config=config,
    )

    assert plan.offspring[0] == 2
    assert plan.offspring[1] == 1
    assert plan.elites[0] == (1, 2)
    assert plan.elites[1] == (3,)
