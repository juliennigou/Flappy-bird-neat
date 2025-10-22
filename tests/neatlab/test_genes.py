from __future__ import annotations

import math

import pytest
from neatlab.genes import ConnectionGene, NodeGene, NodeType


def test_node_gene_creation_and_copy() -> None:
    node = NodeGene(id=1, type="hidden", activation="relu")
    assert node.type is NodeType.HIDDEN
    assert node.activation == "relu"

    copied = node.copy(node_id=5, activation="tanh")
    assert copied.id == 5
    assert copied.type is NodeType.HIDDEN
    assert copied.activation == "tanh"
    # Original gene is immutable.
    assert node.id == 1


def test_node_gene_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        NodeGene(id=-1, type=NodeType.INPUT, activation="relu")

    with pytest.raises(ValueError):
        NodeGene(id=0, type="unknown", activation="relu")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        NodeGene(id=0, type=NodeType.INPUT, activation=" ")


def test_node_type_coerce_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        NodeType.coerce(123)  # type: ignore[arg-type]


def test_connection_gene_creation_and_copy() -> None:
    connection = ConnectionGene(
        innovation=10, in_node_id=1, out_node_id=2, weight=0.5, enabled=True
    )
    assert math.isclose(connection.weight, 0.5)

    updated = connection.copy(weight=1.25, enabled=False)
    assert updated.innovation == connection.innovation
    assert math.isclose(updated.weight, 1.25)
    assert not updated.enabled
    assert connection.enabled  # original unchanged


def test_connection_gene_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        ConnectionGene(innovation=-1, in_node_id=0, out_node_id=1, weight=0.0)

    with pytest.raises(ValueError):
        ConnectionGene(innovation=0, in_node_id=-2, out_node_id=1, weight=0.0)

    with pytest.raises(ValueError):
        ConnectionGene(innovation=0, in_node_id=1, out_node_id=-3, weight=0.0)

    with pytest.raises(ValueError):
        ConnectionGene(innovation=0, in_node_id=1, out_node_id=2, weight="nan")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        ConnectionGene(innovation=0, in_node_id=1, out_node_id=2, weight=object())


def test_connection_gene_toggle() -> None:
    connection = ConnectionGene(innovation=1, in_node_id=2, out_node_id=3, weight=1.0)
    toggled = connection.toggled()

    assert not toggled.enabled
    assert toggled.innovation == connection.innovation
    assert toggled.in_node_id == connection.in_node_id
    assert toggled.out_node_id == connection.out_node_id
