"""Feed-forward network construction and evaluation from NEAT genomes."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .genes import ConnectionGene, NodeGene, NodeType

if TYPE_CHECKING:
    from .genome import Genome

ActivationFunction = Callable[[float], float]
ActivationMap = Mapping[str, ActivationFunction]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


DEFAULT_ACTIVATIONS: dict[str, ActivationFunction] = {
    "identity": lambda x: x,
    "sigmoid": _sigmoid,
    "tanh": math.tanh,
    "relu": lambda x: x if x > 0.0 else 0.0,
}


def compute_feedforward_layers(
    nodes: Mapping[int, NodeGene],
    connections: Iterable[ConnectionGene],
) -> tuple[tuple[int, ...], ...]:
    """Compute topological layers for feed-forward execution."""
    indegree: dict[int, int] = {}
    for node_id in nodes:
        indegree[node_id] = 0
    outgoing: dict[int, list[int]] = defaultdict(list)

    for connection in connections:
        if not connection.enabled:
            continue
        indegree[connection.out_node_id] += 1
        outgoing[connection.in_node_id].append(connection.out_node_id)

    roots = [node_id for node_id, degree in indegree.items() if degree == 0]
    queue: deque[int] = deque(sorted(roots))
    processed: set[int] = set()
    layers: list[tuple[int, ...]] = []

    while queue:
        current_layer: list[int] = []
        next_queue: set[int] = set()
        while queue:
            node_id = queue.popleft()
            if node_id in processed:
                continue
            processed.add(node_id)
            current_layer.append(node_id)
            for target in outgoing.get(node_id, []):
                indegree[target] -= 1
                if indegree[target] == 0:
                    next_queue.add(target)
        if current_layer:
            layers.append(tuple(sorted(current_layer)))
        queue.extend(sorted(next_queue))

    if len(processed) != len(nodes):
        msg = "Cycle detected while computing feed-forward layers."
        raise ValueError(msg)

    return tuple(layers)


def _normalize_activation_name(name: str) -> str:
    return name.strip().lower()


@dataclass(slots=True)
class FeedForwardNetwork:
    """Executable feed-forward network built from a NEAT genome."""

    input_ids: tuple[int, ...]
    bias_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    layers: tuple[tuple[int, ...], ...]
    incoming: Mapping[int, tuple[tuple[int, float], ...]]
    activations: Mapping[int, str]
    activation_functions: dict[str, ActivationFunction]

    @classmethod
    def from_genome(
        cls,
        genome: Genome,
        *,
        activation_functions: ActivationMap | None = None,
    ) -> FeedForwardNetwork:
        """Construct a feed-forward network from a genome."""
        activation_lookup = (
            dict(DEFAULT_ACTIVATIONS)
            if activation_functions is None
            else {name.lower(): fn for name, fn in activation_functions.items()}
        )

        nodes = genome.nodes
        connections = [conn for conn in genome.connections.values() if conn.enabled]

        layers = compute_feedforward_layers(nodes, connections)
        input_ids = tuple(
            sorted(
                node_id
                for node_id, node in nodes.items()
                if node.type is NodeType.INPUT
            )
        )
        bias_ids = tuple(
            sorted(
                node_id for node_id, node in nodes.items() if node.type is NodeType.BIAS
            )
        )
        output_ids = tuple(
            sorted(
                node_id
                for node_id, node in nodes.items()
                if node.type is NodeType.OUTPUT
            )
        )

        incoming: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for connection in connections:
            incoming[connection.out_node_id].append(
                (connection.in_node_id, connection.weight)
            )

        for targets in incoming.values():
            targets.sort(key=lambda item: item[0])

        activation_assignments: dict[int, str] = {
            node_id: _normalize_activation_name(node.activation)
            for node_id, node in nodes.items()
        }

        return cls(
            input_ids=input_ids,
            bias_ids=bias_ids,
            output_ids=output_ids,
            layers=layers,
            incoming={node_id: tuple(targets) for node_id, targets in incoming.items()},
            activations=activation_assignments,
            activation_functions=activation_lookup,
        )

    def activate(self, inputs: Sequence[float]) -> list[float]:
        """Run a forward pass and return outputs in node-id order."""
        if len(inputs) != len(self.input_ids):
            msg = (
                f"Expected {len(self.input_ids)} inputs " f"but received {len(inputs)}."
            )
            raise ValueError(msg)

        values: dict[int, float] = {}
        for bias_id in self.bias_ids:
            values[bias_id] = 1.0
        for node_id, value in zip(self.input_ids, inputs, strict=True):
            values[node_id] = value

        for layer in self.layers:
            for node_id in layer:
                if node_id in values:
                    continue
                incoming_edges = self.incoming.get(node_id, ())
                total = 0.0
                for src_id, weight in incoming_edges:
                    try:
                        src_value = values[src_id]
                    except KeyError as error:
                        msg = f"Missing value for node {src_id} required by {node_id}."
                        raise RuntimeError(msg) from error
                    total += src_value * weight

                activation_name = self.activations[node_id]
                if node_id in self.bias_ids:
                    values[node_id] = 1.0
                    continue
                if node_id in self.input_ids:
                    values[node_id] = total
                    continue

                function = self.activation_functions.get(activation_name)
                if function is None:
                    msg = f"Unknown activation function: {activation_name!r}"
                    raise ValueError(msg)
                values[node_id] = function(total)

        return [values[node_id] for node_id in self.output_ids]


__all__ = [
    "ActivationFunction",
    "ActivationMap",
    "DEFAULT_ACTIVATIONS",
    "FeedForwardNetwork",
    "compute_feedforward_layers",
]
