from __future__ import annotations

import math
from random import Random
from statistics import mean, pstdev

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


def build_minimal_genome() -> Genome:
    nodes = {
        0: NodeGene(id=0, type=NodeType.INPUT, activation="identity"),
        1: NodeGene(id=1, type=NodeType.OUTPUT, activation="identity"),
    }
    connections = {}
    return Genome(nodes=nodes, connections=connections)


def test_mutate_weight_statistics() -> None:
    rng = Random(0)
    nodes = {
        0: NodeGene(id=0, type=NodeType.INPUT, activation="identity"),
        **{
            i: NodeGene(id=i, type=NodeType.HIDDEN, activation="tanh")
            for i in range(1, 101)
        },
    }
    connections = {
        i: ConnectionGene(innovation=i, in_node_id=0, out_node_id=i, weight=0.0)
        for i in range(1, 101)
    }
    genome = Genome(nodes=nodes, connections=connections)

    config = WeightMutationConfig(
        mutate_rate=1.0,
        perturb_sd=0.5,
        reset_rate=0.0,
    )
    mutated = genome.mutate_weight(rng, config)

    weights = [conn.weight for conn in genome.connections.values()]
    assert mutated == len(weights)
    assert abs(mean(weights)) < 0.1
    assert math.isclose(pstdev(weights), config.perturb_sd, rel_tol=0.25)


def test_mutate_weight_reset() -> None:
    rng = Random(1)
    genome = build_minimal_genome()
    innovation = 0
    genome.add_connection(
        ConnectionGene(innovation=innovation, in_node_id=0, out_node_id=1, weight=0.3)
    )

    def constant_init(random: Random) -> float:
        return 0.75

    config = WeightMutationConfig(
        mutate_rate=1.0,
        perturb_sd=0.5,
        reset_rate=1.0,
        weight_init=constant_init,
    )
    genome.mutate_weight(rng, config)

    assert genome.connections[innovation].weight == pytest.approx(0.75)


def test_mutate_weight_respects_rate() -> None:
    rng = Random(2)
    genome = build_minimal_genome()
    genome.add_connection(
        ConnectionGene(innovation=0, in_node_id=0, out_node_id=1, weight=0.9)
    )
    config = WeightMutationConfig(
        mutate_rate=0.0,
        perturb_sd=0.5,
        reset_rate=0.5,
    )
    mutated = genome.mutate_weight(rng, config)

    assert mutated == 0
    assert genome.connections[0].weight == pytest.approx(0.9)


def test_mutate_add_connection_adds_new_edge() -> None:
    genome = build_minimal_genome()
    tracker = InnovationTracker()

    def zero_init(random: Random) -> float:
        return 0.0

    config = AddConnectionConfig(weight_init=zero_init)
    rng = Random(3)
    added = genome.mutate_add_connection(rng, tracker, config)

    assert added is True
    assert genome.contains_connection(0, 1)
    assert genome.connections[0].weight == pytest.approx(0.0)


def test_mutate_add_connection_prevents_cycles() -> None:
    rng = Random(4)
    tracker = InnovationTracker()
    nodes = {
        0: NodeGene(id=0, type=NodeType.INPUT, activation="id"),
        1: NodeGene(id=1, type=NodeType.HIDDEN, activation="tanh"),
        2: NodeGene(id=2, type=NodeType.HIDDEN, activation="relu"),
        3: NodeGene(id=3, type=NodeType.OUTPUT, activation="sigmoid"),
    }
    connections = {
        0: ConnectionGene(0, 0, 1, 0.5),
        1: ConnectionGene(1, 0, 2, -0.3),
        2: ConnectionGene(2, 0, 3, 0.8),
        3: ConnectionGene(3, 1, 2, 0.4),
        4: ConnectionGene(4, 1, 3, 1.2),
        5: ConnectionGene(5, 2, 3, -0.7),
    }
    genome = Genome(nodes=nodes, connections=connections)

    added = genome.mutate_add_connection(rng, tracker, AddConnectionConfig())
    assert added is False
    assert not genome.contains_connection(2, 1)


def test_mutate_add_connection_skips_duplicate() -> None:
    genome = build_minimal_genome()
    tracker = InnovationTracker()
    genome.add_connection(
        ConnectionGene(
            innovation=tracker.register(0, 1),
            in_node_id=0,
            out_node_id=1,
            weight=0.2,
        )
    )

    added = genome.mutate_add_connection(Random(5), tracker, AddConnectionConfig())
    assert added is False


def test_mutate_add_node_splits_connection() -> None:
    genome = build_minimal_genome()
    tracker = InnovationTracker()
    innovation = tracker.register(0, 1)
    genome.add_connection(
        ConnectionGene(innovation=innovation, in_node_id=0, out_node_id=1, weight=0.75)
    )

    added = genome.mutate_add_node(Random(6), tracker, AddNodeConfig(activation="relu"))
    assert added is True

    disabled = genome.connections[innovation]
    assert disabled.enabled is False

    new_node_id = max(genome.nodes)
    new_node = genome.nodes[new_node_id]
    assert new_node.type is NodeType.HIDDEN
    assert new_node.activation == "relu"

    incoming = genome.connections[tracker.register(0, new_node_id)]
    outgoing = genome.connections[tracker.register(new_node_id, 1)]

    assert incoming.weight == pytest.approx(1.0)
    assert outgoing.weight == pytest.approx(0.75)


def test_mutate_add_node_requires_enabled_connection() -> None:
    genome = build_minimal_genome()
    tracker = InnovationTracker()
    innovation = tracker.register(0, 1)
    genome.add_connection(
        ConnectionGene(
            innovation=innovation,
            in_node_id=0,
            out_node_id=1,
            weight=0.5,
            enabled=False,
        )
    )

    result = genome.mutate_add_node(
        Random(7),
        tracker,
        AddNodeConfig(activation="tanh"),
    )
    assert result is False


def _genome_for_crossover() -> tuple[Genome, Genome]:
    nodes = {
        0: NodeGene(0, NodeType.INPUT, "id"),
        1: NodeGene(1, NodeType.HIDDEN, "tanh"),
        2: NodeGene(2, NodeType.OUTPUT, "sigmoid"),
    }
    genome_a = Genome(
        nodes=nodes.copy(),
        connections={
            0: ConnectionGene(0, 0, 1, 0.5, True),
            1: ConnectionGene(1, 1, 2, 1.0, True),
        },
    )
    genome_b = Genome(
        nodes=nodes.copy(),
        connections={
            0: ConnectionGene(0, 0, 1, -0.75, False),
            2: ConnectionGene(2, 0, 2, 0.25, True),
        },
    )
    return genome_a, genome_b


def test_crossover_prefers_fitter_parent() -> None:
    genome_a, genome_b = _genome_for_crossover()
    rng = Random(8)
    child = genome_a.crossover(
        genome_b,
        rng=rng,
        fitness_self=2.0,
        fitness_other=1.0,
        config=CrossoverConfig(disable_inherit_rate=0.5),
    )

    assert set(child.connections) == {0, 1}
    shared = child.connections[0]
    # Deterministic weight pick with rng seed 8 chooses the follower weight.
    assert shared.weight == pytest.approx(-0.75)
    # Only one parent had the gene enabled; probability below 0.5 keeps it enabled.
    assert shared.enabled is True

    assert child.connections[1].weight == pytest.approx(1.0)


def test_crossover_tie_breaks_with_rng() -> None:
    genome_a, genome_b = _genome_for_crossover()
    rng = Random(2)
    child = genome_a.crossover(
        genome_b,
        rng=rng,
        fitness_self=1.0,
        fitness_other=1.0,
    )

    # With seed=2 the random draw selects genome_b as leader.
    assert set(child.connections) == {0, 2}
    assert child.connections[2].weight == pytest.approx(0.25)
    assert child.connections[0].enabled is False
