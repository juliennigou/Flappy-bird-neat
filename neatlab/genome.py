"""Genome representation and mutation/crossover operators."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from random import Random

from .genes import ConnectionGene, NodeGene, NodeType
from .innovations import InnovationTracker

WeightInitializer = Callable[[Random], float]


def _default_weight_init(rng: Random) -> float:
    return rng.uniform(-1.0, 1.0)


@dataclass(frozen=True, slots=True)
class WeightMutationConfig:
    """Configuration for weight mutation behaviour."""

    mutate_rate: float
    perturb_sd: float
    reset_rate: float
    weight_init: WeightInitializer = _default_weight_init

    def __post_init__(self) -> None:
        for label, value in (
            ("mutate_rate", self.mutate_rate),
            ("reset_rate", self.reset_rate),
        ):
            if not 0.0 <= value <= 1.0:
                msg = f"{label} must be in [0, 1]."
                raise ValueError(msg)
        if self.perturb_sd <= 0.0:
            msg = "perturb_sd must be positive."
            raise ValueError(msg)
        if self.weight_init is None:
            msg = "weight_init callable must be provided."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class AddConnectionConfig:
    """Configuration for add-connection mutation."""

    allow_recurrent: bool = False
    max_attempts: int = 32
    weight_init: WeightInitializer = _default_weight_init

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            msg = "max_attempts must be positive."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class AddNodeConfig:
    """Configuration for add-node mutation."""

    activation: str

    def __post_init__(self) -> None:
        if not self.activation or not self.activation.strip():
            msg = "activation must be a non-empty string."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CrossoverConfig:
    """Configuration controlling crossover behaviour."""

    disable_inherit_rate: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.disable_inherit_rate <= 1.0:
            msg = "disable_inherit_rate must be in [0, 1]."
            raise ValueError(msg)


@dataclass(slots=True)
class Genome:
    """NEAT genome containing nodes and connection genes."""

    nodes: dict[int, NodeGene]
    connections: dict[int, ConnectionGene]
    _pair_index: dict[tuple[int, int], int] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _next_node_id: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.nodes:
            msg = "Genome must contain at least one node."
            raise ValueError(msg)
        self._next_node_id = max(self.nodes) + 1
        for innovation, connection in self.connections.items():
            if connection.innovation != innovation:
                msg = (
                    f"Connection innovation mismatch: key {innovation} "
                    f"!= connection.innovation {connection.innovation}"
                )
                raise ValueError(msg)
            if (
                connection.in_node_id not in self.nodes
                or connection.out_node_id not in self.nodes
            ):
                msg = "Connection references unknown node."
                raise ValueError(msg)
            pair = (connection.in_node_id, connection.out_node_id)
            if pair in self._pair_index:
                msg = f"Duplicate connection between nodes {pair}."
                raise ValueError(msg)
            self._pair_index[pair] = innovation

    def copy(self) -> Genome:
        """Return a shallow copy of the genome."""
        nodes = {node_id: gene.copy() for node_id, gene in self.nodes.items()}
        connections = {
            innovation: conn.copy() for innovation, conn in self.connections.items()
        }
        return Genome(nodes=nodes, connections=connections)

    def contains_connection(self, in_node: int, out_node: int) -> bool:
        """Return whether a connection between the nodes exists."""
        return (in_node, out_node) in self._pair_index

    def add_connection(self, connection: ConnectionGene) -> None:
        """Add a new connection gene to the genome."""
        pair = (connection.in_node_id, connection.out_node_id)
        if pair in self._pair_index:
            msg = f"Connection between {pair} already exists."
            raise ValueError(msg)
        if connection.innovation in self.connections:
            msg = f"Connection innovation {connection.innovation} already present."
            raise ValueError(msg)
        if (
            connection.in_node_id not in self.nodes
            or connection.out_node_id not in self.nodes
        ):
            msg = "Connection references unknown node."
            raise ValueError(msg)
        self.connections[connection.innovation] = connection
        self._pair_index[pair] = connection.innovation

    def add_node(self, node: NodeGene) -> None:
        """Register a new node gene in the genome."""
        if node.id in self.nodes:
            msg = f"Node {node.id} already exists."
            raise ValueError(msg)
        self.nodes[node.id] = node
        self._next_node_id = max(self._next_node_id, node.id + 1)

    def allocate_node_id(self) -> int:
        """Allocate a fresh node identifier."""
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def mutate_weight(self, rng: Random, config: WeightMutationConfig) -> int:
        """Mutate connection weights in-place.

        Returns:
            The number of connections that had their weight modified.
        """
        mutated = 0
        for innovation, connection in list(self.connections.items()):
            if rng.random() > config.mutate_rate:
                continue
            mutated += 1
            if rng.random() < config.reset_rate:
                new_weight = config.weight_init(rng)
                updated = connection.copy(weight=new_weight)
            else:
                delta = rng.gauss(0.0, config.perturb_sd)
                updated = connection.copy(weight=connection.weight + delta)
            self.connections[innovation] = updated
        return mutated

    def mutate_add_connection(
        self,
        rng: Random,
        tracker: InnovationTracker,
        config: AddConnectionConfig,
    ) -> bool:
        """Add a new connection gene if a valid candidate exists."""
        candidates = list(self._iter_connection_candidates(config.allow_recurrent))
        if not candidates:
            return False

        attempts = min(config.max_attempts, len(candidates))
        for in_id, out_id in rng.sample(candidates, k=attempts):
            if self.contains_connection(in_id, out_id):
                continue
            if not config.allow_recurrent and self._introduces_cycle(in_id, out_id):
                continue
            innovation = tracker.register(in_id, out_id)
            weight = config.weight_init(rng)
            new_conn = ConnectionGene(
                innovation=innovation,
                in_node_id=in_id,
                out_node_id=out_id,
                weight=weight,
                enabled=True,
            )
            self.add_connection(new_conn)
            return True
        return False

    def mutate_add_node(
        self,
        rng: Random,
        tracker: InnovationTracker,
        config: AddNodeConfig,
    ) -> bool:
        """Split an enabled connection by inserting a hidden node."""
        enabled_connections = [
            connection for connection in self.connections.values() if connection.enabled
        ]
        if not enabled_connections:
            return False
        connection = rng.choice(enabled_connections)
        disabled = connection.copy(enabled=False)
        self.connections[connection.innovation] = disabled

        new_node_id = self.allocate_node_id()
        new_node = NodeGene(
            id=new_node_id,
            type=NodeType.HIDDEN,
            activation=config.activation,
        )
        self.add_node(new_node)

        in_innovation = tracker.register(connection.in_node_id, new_node_id)
        out_innovation = tracker.register(new_node_id, connection.out_node_id)

        incoming = ConnectionGene(
            innovation=in_innovation,
            in_node_id=connection.in_node_id,
            out_node_id=new_node_id,
            weight=1.0,
            enabled=True,
        )
        outgoing = ConnectionGene(
            innovation=out_innovation,
            in_node_id=new_node_id,
            out_node_id=connection.out_node_id,
            weight=connection.weight,
            enabled=True,
        )
        self.add_connection(incoming)
        self.add_connection(outgoing)
        return True

    def crossover(
        self,
        other: Genome,
        *,
        rng: Random,
        fitness_self: float,
        fitness_other: float,
        config: CrossoverConfig | None = None,
    ) -> Genome:
        """Create a child genome via crossover.

        Args:
            other: The second parent genome.
            rng: Random generator controlling stochastic choices.
            fitness_self: Fitness score of this genome.
            fitness_other: Fitness score of the other genome.
            config: Behavioural configuration (optional).
        """
        if config is None:
            config = CrossoverConfig()

        if fitness_self > fitness_other:
            leader, follower = self, other
        elif fitness_self < fitness_other:
            leader, follower = other, self
        else:
            if rng.random() < 0.5:
                leader, follower = self, other
            else:
                leader, follower = other, self

        follower_conns = follower.connections
        child_connections: dict[int, ConnectionGene] = {}

        for innovation, lead_conn in leader.connections.items():
            follower_conn = follower_conns.get(innovation)
            if follower_conn is None:
                child_connections[innovation] = lead_conn.copy()
                continue

            picked_weight = lead_conn.weight
            if rng.random() < 0.5:
                picked_weight = follower_conn.weight

            enabled = lead_conn.enabled or follower_conn.enabled
            if not (lead_conn.enabled and follower_conn.enabled):
                if rng.random() < config.disable_inherit_rate:
                    enabled = False

            merged = lead_conn.copy(weight=picked_weight, enabled=enabled)
            child_connections[innovation] = merged

        child_nodes: dict[int, NodeGene] = {}
        for node_map in (leader.nodes, follower.nodes):
            for node_id, node in node_map.items():
                child_nodes.setdefault(node_id, node.copy())

        return Genome(nodes=child_nodes, connections=child_connections)

    def _iter_connection_candidates(
        self,
        allow_recurrent: bool,
    ) -> Iterator[tuple[int, int]]:
        """Yield potential connection pairs respecting node type constraints."""
        valid_inputs = [
            node_id
            for node_id, gene in self.nodes.items()
            if gene.type in (NodeType.INPUT, NodeType.HIDDEN, NodeType.BIAS)
        ]
        valid_outputs = [
            node_id
            for node_id, gene in self.nodes.items()
            if gene.type in (NodeType.HIDDEN, NodeType.OUTPUT)
        ]
        for in_id in valid_inputs:
            for out_id in valid_outputs:
                if in_id == out_id:
                    continue
                yield (in_id, out_id)

    def _introduces_cycle(self, in_id: int, out_id: int) -> bool:
        """Detect whether adding an edge would create a cycle."""
        stack = [out_id]
        visited: set[int] = set()
        while stack:
            current = stack.pop()
            if current == in_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            for connection in self.connections.values():
                if not connection.enabled:
                    continue
                if connection.in_node_id == current:
                    stack.append(connection.out_node_id)
        return False


__all__ = [
    "AddConnectionConfig",
    "AddNodeConfig",
    "CrossoverConfig",
    "Genome",
    "WeightMutationConfig",
]
