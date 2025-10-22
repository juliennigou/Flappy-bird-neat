from __future__ import annotations

import math

import pytest
from neatlab.genes import ConnectionGene, NodeGene, NodeType
from neatlab.genome import Genome
from neatlab.network import (
    DEFAULT_ACTIVATIONS,
    FeedForwardNetwork,
    compute_feedforward_layers,
)


def _build_nodes(*definitions: tuple[int, NodeType, str]) -> dict[int, NodeGene]:
    return {
        node_id: NodeGene(id=node_id, type=node_type, activation=activation)
        for node_id, node_type, activation in definitions
    }


def test_compute_layers_orders_nodes_by_dependencies() -> None:
    nodes = _build_nodes(
        (0, NodeType.INPUT, "identity"),
        (1, NodeType.BIAS, "identity"),
        (2, NodeType.HIDDEN, "relu"),
        (3, NodeType.OUTPUT, "sigmoid"),
    )
    connections = [
        ConnectionGene(innovation=0, in_node_id=0, out_node_id=2, weight=0.5),
        ConnectionGene(innovation=1, in_node_id=1, out_node_id=2, weight=1.0),
        ConnectionGene(innovation=2, in_node_id=2, out_node_id=3, weight=1.0),
    ]

    layers = compute_feedforward_layers(nodes, connections)
    assert layers == ((0, 1), (2,), (3,))


def test_compute_layers_raises_on_cycle() -> None:
    nodes = _build_nodes(
        (0, NodeType.INPUT, "identity"),
        (1, NodeType.HIDDEN, "tanh"),
    )
    connections = [
        ConnectionGene(innovation=0, in_node_id=0, out_node_id=1, weight=0.5),
        ConnectionGene(innovation=1, in_node_id=1, out_node_id=0, weight=0.5),
    ]

    with pytest.raises(ValueError):
        compute_feedforward_layers(nodes, connections)


def test_feedforward_network_evaluates_outputs() -> None:
    nodes = _build_nodes(
        (0, NodeType.INPUT, "identity"),
        (1, NodeType.INPUT, "identity"),
        (2, NodeType.BIAS, "identity"),
        (3, NodeType.HIDDEN, "relu"),
        (4, NodeType.OUTPUT, "sigmoid"),
    )
    connections = {
        0: ConnectionGene(0, 0, 3, 0.5),
        1: ConnectionGene(1, 1, 3, 0.4),
        2: ConnectionGene(2, 2, 3, 0.1),
        3: ConnectionGene(3, 3, 4, 1.2),
        4: ConnectionGene(4, 2, 4, -0.3),
    }
    genome = Genome(nodes=nodes, connections=connections)

    network = FeedForwardNetwork.from_genome(genome)

    outputs = network.activate([0.2, -0.4])
    assert len(outputs) == 1
    expected_hidden = max(0.0, 0.5 * 0.2 + 0.4 * -0.4 + 0.1 * 1.0)
    expected_output = DEFAULT_ACTIVATIONS["sigmoid"](1.2 * expected_hidden + -0.3 * 1.0)
    assert math.isclose(expected_hidden, 0.04, rel_tol=1e-6)
    assert outputs[0] == pytest.approx(expected_output)


def test_feedforward_unknown_activation_raises() -> None:
    nodes = _build_nodes(
        (0, NodeType.INPUT, "identity"),
        (1, NodeType.OUTPUT, "mystery"),
    )
    connections = {0: ConnectionGene(0, 0, 1, 0.5)}
    genome = Genome(nodes=nodes, connections=connections)

    network = FeedForwardNetwork.from_genome(genome)
    with pytest.raises(ValueError):
        network.activate([1.0])


def test_feedforward_respects_disabled_connections() -> None:
    nodes = _build_nodes(
        (0, NodeType.INPUT, "identity"),
        (1, NodeType.HIDDEN, "identity"),
        (2, NodeType.OUTPUT, "identity"),
    )
    connections = {
        0: ConnectionGene(0, 0, 1, 1.0, enabled=False),
        1: ConnectionGene(1, 0, 2, 2.0, enabled=True),
    }
    genome = Genome(nodes=nodes, connections=connections)
    network = FeedForwardNetwork.from_genome(genome)

    outputs = network.activate([3.0])
    assert outputs == [6.0]
