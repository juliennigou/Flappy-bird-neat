from __future__ import annotations

from random import Random

import pytest
from neatlab.genes import ConnectionGene, NodeGene, NodeType
from neatlab.genome import Genome
from neatlab.species import (
    SpeciesConfig,
    SpeciesManager,
    compatibility_distance,
)


def build_genome(connection_defs: list[tuple[int, int, int, float]]) -> Genome:
    nodes = {
        0: NodeGene(0, NodeType.INPUT, "identity"),
        1: NodeGene(1, NodeType.HIDDEN, "tanh"),
        2: NodeGene(2, NodeType.OUTPUT, "identity"),
    }
    connections = {
        innovation: ConnectionGene(
            innovation=innovation,
            in_node_id=in_id,
            out_node_id=out_id,
            weight=weight,
        )
        for innovation, in_id, out_id, weight in connection_defs
    }
    return Genome(nodes=nodes, connections=connections)


def test_compatibility_distance_follows_formula() -> None:
    genome_a = build_genome(
        [
            (0, 0, 1, 0.5),
            (1, 1, 2, 0.7),
        ]
    )
    genome_b = build_genome(
        [
            (0, 0, 1, 0.6),
            (2, 1, 2, 0.9),
        ]
    )

    delta = compatibility_distance(genome_a, genome_b, c1=1.0, c2=1.0, c3=0.4)
    assert pytest.approx(2.04, rel=1e-6) == delta

    # Symmetry check.
    delta_reverse = compatibility_distance(genome_b, genome_a, c1=1.0, c2=1.0, c3=0.4)
    assert delta_reverse == pytest.approx(delta)


def test_speciation_groups_by_threshold_and_updates_fitness() -> None:
    config = SpeciesConfig(
        c1=1.0,
        c2=1.0,
        c3=0.4,
        compatibility_threshold=2.5,
        target_species=3,
        species_adjust_period=1,
        adjust_rate=0.5,
    )
    manager = SpeciesManager(config)

    genomes = {
        1: build_genome([(0, 0, 1, 0.5), (1, 1, 2, 0.7)]),
        2: build_genome([(0, 0, 1, 0.52), (1, 1, 2, 0.69)]),
        3: build_genome([(0, 0, 1, -0.9), (3, 1, 2, 1.4)]),
    }
    fitnesses = {1: 1.0, 2: 1.5, 3: 2.2}

    species = manager.speciate(
        genomes,
        generation=0,
        fitnesses=fitnesses,
        rng=Random(0),
    )

    assert len(species) == 2
    assert species[0].members == [1, 2]
    assert species[1].members == [3]
    assert species[0].best_fitness == pytest.approx(1.5)
    assert species[0].last_improved_generation == 0
    assert species[1].age == 0


def test_adjust_threshold_moves_towards_target() -> None:
    config = SpeciesConfig(
        c1=1.0,
        c2=1.0,
        c3=0.4,
        compatibility_threshold=3.0,
        target_species=4,
        species_adjust_period=2,
        adjust_rate=0.5,
        min_compatibility_threshold=1.0,
        max_compatibility_threshold=5.0,
    )
    manager = SpeciesManager(config)

    manager.adjust_threshold(generation=1, species_count=2)
    assert manager.compatibility_threshold == pytest.approx(3.0)

    manager.adjust_threshold(generation=2, species_count=6)
    assert manager.compatibility_threshold == pytest.approx(3.5)

    manager.adjust_threshold(generation=4, species_count=2)
    assert manager.compatibility_threshold == pytest.approx(3.0)


def test_species_config_validation_errors() -> None:
    with pytest.raises(ValueError):
        SpeciesConfig(
            c1=-1.0,
            c2=1.0,
            c3=1.0,
            compatibility_threshold=1.0,
            target_species=2,
            species_adjust_period=1,
        )
    with pytest.raises(ValueError):
        SpeciesConfig(
            c1=1.0,
            c2=1.0,
            c3=1.0,
            compatibility_threshold=-0.5,
            target_species=2,
            species_adjust_period=1,
        )
    with pytest.raises(ValueError):
        SpeciesConfig(
            c1=1.0,
            c2=1.0,
            c3=1.0,
            compatibility_threshold=1.0,
            target_species=0,
            species_adjust_period=1,
        )
    with pytest.raises(ValueError):
        SpeciesConfig(
            c1=1.0,
            c2=1.0,
            c3=1.0,
            compatibility_threshold=1.0,
            target_species=1,
            species_adjust_period=1,
            adjust_rate=0.0,
        )
    with pytest.raises(ValueError):
        SpeciesConfig(
            c1=1.0,
            c2=1.0,
            c3=1.0,
            compatibility_threshold=1.0,
            target_species=1,
            species_adjust_period=1,
            min_compatibility_threshold=2.0,
            max_compatibility_threshold=1.0,
        )


def test_speciate_requires_fitness_entries() -> None:
    config = SpeciesConfig(
        c1=1.0,
        c2=1.0,
        c3=0.4,
        compatibility_threshold=2.5,
        target_species=3,
        species_adjust_period=1,
    )
    manager = SpeciesManager(config)
    genomes = {1: build_genome([(0, 0, 1, 0.5)])}

    with pytest.raises(KeyError):
        manager.speciate(genomes, generation=0, fitnesses={}, rng=Random(1))


def test_speciate_clears_when_no_genomes() -> None:
    config = SpeciesConfig(
        c1=1.0,
        c2=1.0,
        c3=0.4,
        compatibility_threshold=2.5,
        target_species=3,
        species_adjust_period=1,
    )
    manager = SpeciesManager(config)
    genomes = {1: build_genome([(0, 0, 1, 0.5)])}
    manager.speciate(genomes, generation=0, rng=Random(0), fitnesses={1: 1.0})

    result = manager.speciate({}, generation=1, rng=Random(1))
    assert result == ()
    assert manager.species() == ()
