"""Benchmark helpers to measure evaluator throughput."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from time import perf_counter
from typing import Sequence

from games.flappy.env.adapter import action_transform, obs_transform
from games.flappy.env.env_core import FlappyEnv, FlappyEnvConfig

from .evaluator import EvaluationConfig, ParallelEvaluator, SyncEvaluator
from .genes import ConnectionGene, NodeGene, NodeType
from .genome import Genome


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Summary of a single benchmark measurement."""

    workers: int
    steps: int
    duration_s: float
    steps_per_sec: float
    iterations: int


def _baseline_population(size: int, rng: Random) -> dict[int, Genome]:
    nodes = {
        0: NodeGene(0, NodeType.INPUT, "identity"),
        1: NodeGene(1, NodeType.INPUT, "identity"),
        2: NodeGene(2, NodeType.INPUT, "identity"),
        3: NodeGene(3, NodeType.INPUT, "identity"),
        4: NodeGene(4, NodeType.BIAS, "identity"),
        5: NodeGene(5, NodeType.OUTPUT, "sigmoid"),
    }
    population: dict[int, Genome] = {}
    for genome_id in range(size):
        connections = {
            idx: ConnectionGene(idx, in_id, 5, rng.uniform(-1.0, 1.0))
            for idx, in_id in enumerate((0, 1, 2, 3, 4))
        }
        population[genome_id] = Genome(nodes=nodes.copy(), connections=connections)
    return population


def run_benchmark(
    env_config: FlappyEnvConfig,
    *,
    steps_target: int,
    population_size: int,
    episodes_per_genome: int,
    worker_counts: Sequence[int],
    batch_size: int,
    seed: int | None = None,
) -> list[BenchmarkResult]:
    """Execute benchmarks for the provided worker counts."""
    if steps_target <= 0:
        msg = "steps_target must be positive."
        raise ValueError(msg)
    if population_size <= 0:
        msg = "population_size must be positive."
        raise ValueError(msg)
    if episodes_per_genome <= 0:
        msg = "episodes_per_genome must be positive."
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive."
        raise ValueError(msg)
    if not worker_counts:
        msg = "At least one worker count must be provided."
        raise ValueError(msg)

    evaluation_config = EvaluationConfig(
        episodes_per_genome=episodes_per_genome,
        seed=seed,
    )

    results: list[BenchmarkResult] = []
    for workers in worker_counts:
        if workers <= 0:
            msg = "Worker counts must be positive integers."
            raise ValueError(msg)

        rng = Random(seed if seed is not None else 0)
        genomes = _baseline_population(population_size, rng)

        if workers == 1:
            evaluator: SyncEvaluator | ParallelEvaluator = SyncEvaluator(
                lambda: FlappyEnv(env_config),
                obs_transform,
                action_transform,
                config=evaluation_config,
            )
        else:
            evaluator = ParallelEvaluator(
                lambda: FlappyEnv(env_config),
                obs_transform,
                action_transform,
                workers=workers,
                batch_size=batch_size,
                config=evaluation_config,
            )

        steps_accum = 0
        iterations = 0
        start = perf_counter()
        while steps_accum < steps_target:
            evaluator(genomes, rng)
            stats = evaluator.last_stats
            if stats.steps <= 0:
                break
            steps_accum += stats.steps
            iterations += 1
        duration = perf_counter() - start
        throughput = (
            steps_accum / duration if steps_accum > 0 and duration > 0.0 else 0.0
        )
        results.append(
            BenchmarkResult(
                workers=workers,
                steps=steps_accum,
                duration_s=duration,
                steps_per_sec=throughput,
                iterations=iterations,
            )
        )
    return results


__all__ = ["BenchmarkResult", "run_benchmark"]
