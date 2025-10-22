from __future__ import annotations

from random import Random

import pytest
from neatlab.genes import ConnectionGene, NodeGene, NodeType
from neatlab.genome import (
    AddConnectionConfig,
    AddNodeConfig,
    CrossoverConfig,
    Genome,
    WeightMutationConfig,
)
from neatlab.innovations import InnovationTracker
from neatlab.population import MutationOperators, PopulationConfig, PopulationState
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


def _operators(
    *,
    tracker: InnovationTracker,
    weight_value: float = 42.0,
    add_conn_rate: float = 0.0,
    add_node_rate: float = 0.0,
    crossover_rate: float = 0.0,
) -> MutationOperators:
    weight_config = WeightMutationConfig(
        mutate_rate=1.0,
        perturb_sd=0.1,
        reset_rate=1.0,
        weight_init=lambda rng: weight_value,
    )
    return MutationOperators(
        tracker=tracker,
        weight=weight_config,
        add_connection=AddConnectionConfig(allow_recurrent=False, max_attempts=5),
        add_node=AddNodeConfig(activation="tanh"),
        add_connection_rate=add_conn_rate,
        add_node_rate=add_node_rate,
        crossover_rate=crossover_rate,
        crossover=CrossoverConfig(disable_inherit_rate=0.0),
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

    tracker = InnovationTracker()
    operators = _operators(tracker=tracker, weight_value=3.5)

    population.reproduce(species, operators)

    assert population.generation == 1
    assert len(population.genomes) == population.config.population_size
    assert len(set(population.genomes)) == population.config.population_size
    weights = [
        next(iter(genome.connections.values())).weight
        for genome in population.genomes.values()
    ]
    assert any(weight == pytest.approx(3.5) for weight in weights)
    assert population.champion_id is not None


def test_reproduce_fallbacks_when_all_species_stagnant() -> None:
    population = _population_state(pop_size=2)
    population.evaluate(lambda genomes, rng: dict.fromkeys(genomes, 1.0))
    species = population.speciate()

    population.stagnant_species = {item.id for item in species}

    tracker = InnovationTracker()
    operators = _operators(tracker=tracker)

    population.reproduce(species, operators)
    assert population.generation == 1


def test_reproduce_can_add_connection() -> None:
    nodes = {
        0: NodeGene(0, NodeType.INPUT, "identity"),
        1: NodeGene(1, NodeType.INPUT, "identity"),
        2: NodeGene(2, NodeType.OUTPUT, "identity"),
    }
    base_connections = {0: ConnectionGene(0, 0, 2, 1.0)}

    genomes = {
        idx: Genome(
            nodes={nid: gene.copy() for nid, gene in nodes.items()},
            connections={
                innovation: conn.copy()
                for innovation, conn in base_connections.items()
            },
        )
        for idx in range(2)
    }

    population = PopulationState(
        generation=0,
        genomes=genomes,
        rng=Random(0),
        species_manager=_species_manager(),
        config=PopulationConfig(
            population_size=2,
            elitism=0,
            survival_threshold=0.5,
            max_stagnation=1,
        ),
        reproduction_config=ReproductionConfig(elitism=0, survival_threshold=0.5),
    )

    population.evaluate(lambda genomes, rng: {gid: 1.0 for gid in genomes})
    species = population.speciate()

    tracker = InnovationTracker()
    tracker.register(0, 2)
    operators = _operators(
        tracker=tracker,
        weight_value=1.0,
        add_conn_rate=1.0,
        add_node_rate=0.0,
        crossover_rate=0.0,
    )

    population.reproduce(species, operators)

    connection_counts = [len(genome.connections) for genome in population.genomes.values()]
    assert any(count > 1 for count in connection_counts)
    assert tracker.peek(1, 2) is not None


def test_reproduce_falls_back_on_cycle(monkeypatch) -> None:
    nodes = {
        0: NodeGene(0, NodeType.INPUT, "identity"),
        1: NodeGene(1, NodeType.HIDDEN, "tanh"),
        2: NodeGene(2, NodeType.HIDDEN, "tanh"),
        3: NodeGene(3, NodeType.OUTPUT, "identity"),
    }
    connections = {
        0: ConnectionGene(0, 0, 1, 1.0),
        1: ConnectionGene(1, 1, 2, 1.0),
        2: ConnectionGene(2, 2, 3, 1.0),
    }
    genomes = {
        0: Genome(
            nodes={nid: gene.copy() for nid, gene in nodes.items()},
            connections={cid: conn.copy() for cid, conn in connections.items()},
        ),
        1: Genome(
            nodes={nid: gene.copy() for nid, gene in nodes.items()},
            connections={cid: conn.copy() for cid, conn in connections.items()},
        ),
    }

    population = PopulationState(
        generation=0,
        genomes=genomes,
        rng=Random(0),
        species_manager=_species_manager(),
        config=PopulationConfig(
            population_size=2,
            elitism=0,
            survival_threshold=0.5,
            max_stagnation=2,
        ),
        reproduction_config=ReproductionConfig(elitism=0, survival_threshold=1.0),
    )
    population.evaluate(lambda genomes, rng: {gid: 1.0 for gid in genomes})
    species = population.speciate()

    tracker = InnovationTracker()
    for conn in connections.values():
        tracker.register(conn.in_node_id, conn.out_node_id)

    operators = _operators(
        tracker=tracker,
        weight_value=1.0,
        add_conn_rate=1.0,
        add_node_rate=0.0,
        crossover_rate=0.0,
    )

    def force_cycle(
        genome: Genome,
        rng: Random,
        tracker: InnovationTracker,
        config: AddConnectionConfig,
    ) -> bool:
        innovation = tracker.register(2, 1)
        genome.add_connection(
            ConnectionGene(
                innovation=innovation,
                in_node_id=2,
                out_node_id=1,
                weight=1.0,
                enabled=True,
            )
        )
        return True

    monkeypatch.setattr(Genome, "mutate_add_connection", force_cycle)

    population.reproduce(species, operators)

    for genome in population.genomes.values():
        assert all(
            not (conn.in_node_id == 2 and conn.out_node_id == 1)
            for conn in genome.connections.values()
        )
