from __future__ import annotations

from collections.abc import Sequence
from random import Random
from typing import Any

import pytest
from neatlab.evaluator import EvaluationConfig, ParallelEvaluator, SyncEvaluator
from neatlab.genes import ConnectionGene, NodeGene, NodeType
from neatlab.genome import Genome


def _simple_genome(weight: float = 1.0) -> Genome:
    nodes = {
        0: NodeGene(0, NodeType.INPUT, "identity"),
        1: NodeGene(1, NodeType.OUTPUT, "identity"),
    }
    connections = {0: ConnectionGene(0, 0, 1, weight)}
    return Genome(nodes=nodes, connections=connections)


class LoggingEnv:
    def __init__(self, registry: list[int] | None = None) -> None:
        self.registry = registry
        self._seed: int | None = None

    def reset(self, seed: int | None = None) -> float:
        self._seed = 0 if seed is None else seed
        if self.registry is not None:
            self.registry.append(self._seed)
        self._done = False
        return float(self._seed)

    def step(self, action: float) -> tuple[float, float, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step called after episode termination")
        self._done = True
        reward = float(self._seed)
        return float(self._seed), reward, True, {}


def _env_factory(registry: list[int] | None = None) -> LoggingEnv:
    return LoggingEnv(registry)


class FailingEnv(LoggingEnv):
    def reset(self, seed: int | None = None) -> float:
        raise RuntimeError("boom")


def _failing_env_factory() -> FailingEnv:
    return FailingEnv()


def _obs_transform(obs: float) -> Sequence[float]:
    return [float(obs)]


def _action_transform(outputs: Sequence[float]) -> float:
    return float(outputs[0])


def test_sync_evaluator_returns_average_reward() -> None:
    genomes = {0: _simple_genome(), 1: _simple_genome(weight=1.5)}
    evaluator = SyncEvaluator(
        _env_factory,
        _obs_transform,
        _action_transform,
        config=EvaluationConfig(episodes_per_genome=2, seed=100),
    )
    rng = Random(123)

    result = evaluator(genomes, rng)

    assert result == {0: pytest.approx(100.5), 1: pytest.approx(102.5)}


def test_sync_evaluator_is_deterministic_with_seed() -> None:
    genomes = {0: _simple_genome()}
    config = EvaluationConfig(episodes_per_genome=3, seed=42)
    evaluator = SyncEvaluator(
        _env_factory,
        _obs_transform,
        _action_transform,
        config=config,
    )

    rng_a = Random(999)
    rng_b = Random(0)

    result_a = evaluator(genomes, rng_a)
    result_b = evaluator(genomes, rng_b)

    assert result_a == result_b


def test_parallel_evaluator_matches_sync() -> None:
    genomes = {idx: _simple_genome(weight=1.0 + idx * 0.2) for idx in range(4)}
    config = EvaluationConfig(episodes_per_genome=2, seed=7)
    sync_eval = SyncEvaluator(
        _env_factory,
        _obs_transform,
        _action_transform,
        config=config,
    )
    parallel_eval = ParallelEvaluator(
        _env_factory,
        _obs_transform,
        _action_transform,
        workers=1,
        batch_size=1,
        config=config,
    )

    rng = Random(555)
    expected = sync_eval(genomes, Random(555))
    result = parallel_eval(genomes, rng)

    assert result == pytest.approx(expected)


def test_parallel_evaluator_seed_assignment() -> None:
    genomes = {idx: _simple_genome() for idx in range(3)}
    config = EvaluationConfig(episodes_per_genome=2, seed=5)
    evaluator = ParallelEvaluator(
        _env_factory,
        _obs_transform,
        _action_transform,
        workers=1,
        batch_size=2,
        config=config,
    )

    result = evaluator(genomes, Random(123))

    expected = {}
    seed = config.seed or 0
    for genome_id in sorted(genomes):
        seeds = [seed, seed + 1]
        expected[genome_id] = pytest.approx(sum(seeds) / len(seeds))
        seed += config.episodes_per_genome

    assert result == expected


def test_parallel_evaluator_raises_on_worker_failure() -> None:
    genomes = {0: _simple_genome()}
    evaluator = ParallelEvaluator(
        _failing_env_factory,
        _obs_transform,
        _action_transform,
        workers=1,
        batch_size=1,
        config=EvaluationConfig(episodes_per_genome=1, seed=1),
    )

    with pytest.raises(RuntimeError):
        evaluator(genomes, Random(0))
