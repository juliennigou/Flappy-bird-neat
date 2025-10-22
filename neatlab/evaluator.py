"""Evaluators that score genomes against environments."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Any

from .genome import Genome
from .network import FeedForwardNetwork

EnvFactory = Callable[[], Any]
ObsTransform = Callable[[Any], Sequence[float]]
ActionTransform = Callable[[Sequence[float]], Any]


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Configuration shared by evaluators."""

    episodes_per_genome: int = 1
    max_steps: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.episodes_per_genome <= 0:
            msg = "episodes_per_genome must be positive."
            raise ValueError(msg)
        if self.max_steps is not None and self.max_steps <= 0:
            msg = "max_steps must be positive when provided."
            raise ValueError(msg)


def _episode_seed_series(
    base_seed: int,
    worker_id: int,
    episode_counter: int,
    episodes_per_genome: int,
) -> tuple[list[int], int]:
    start = base_seed + worker_id * 10_000 + episode_counter
    seeds = [start + offset for offset in range(episodes_per_genome)]
    return seeds, episode_counter + episodes_per_genome


def _evaluate_genome(
    genome: Genome,
    env_factory: EnvFactory,
    obs_transform: ObsTransform,
    action_transform: ActionTransform,
    config: EvaluationConfig,
    episode_seeds: Iterable[int],
) -> float:
    network = FeedForwardNetwork.from_genome(genome)
    rewards: list[float] = []
    for seed in episode_seeds:
        env = env_factory()
        observation = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        done = False

        while True:
            inputs = obs_transform(observation)
            outputs = network.activate(inputs)
            action = action_transform(outputs)
            observation, reward, done, _info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if done:
                break
            if config.max_steps is not None and steps >= config.max_steps:
                break

        rewards.append(total_reward)
        if hasattr(env, "close"):
            env.close()
    return sum(rewards) / len(rewards)


class SyncEvaluator:
    """Single-process evaluator that runs genomes sequentially."""

    def __init__(
        self,
        env_factory: EnvFactory,
        obs_transform: ObsTransform,
        action_transform: ActionTransform,
        *,
        config: EvaluationConfig | None = None,
    ) -> None:
        self.env_factory = env_factory
        self.obs_transform = obs_transform
        self.action_transform = action_transform
        self.config = config or EvaluationConfig()

    def __call__(self, genomes: Mapping[int, Genome], rng: Random) -> dict[int, float]:
        if not genomes:
            return {}

        base_seed = (
            self.config.seed if self.config.seed is not None else rng.getrandbits(32)
        )
        episode_counter = 0
        results: dict[int, float] = {}

        for genome_id, genome in sorted(genomes.items(), key=lambda item: item[0]):
            seeds, episode_counter = _episode_seed_series(
                base_seed=base_seed,
                worker_id=0,
                episode_counter=episode_counter,
                episodes_per_genome=self.config.episodes_per_genome,
            )
            fitness = _evaluate_genome(
                genome,
                self.env_factory,
                self.obs_transform,
                self.action_transform,
                self.config,
                seeds,
            )
            results[genome_id] = fitness

        return results


def _worker_loop(
    worker_id: int,
    env_factory: EnvFactory,
    obs_transform: ObsTransform,
    action_transform: ActionTransform,
    config: EvaluationConfig,
    base_seed: int,
    task_queue: multiprocessing.queues.Queue[Any],
    result_queue: multiprocessing.queues.Queue[Any],
) -> None:
    episode_counter = 0
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            genome_id, genome = task
            seeds, episode_counter = _episode_seed_series(
                base_seed=base_seed,
                worker_id=worker_id,
                episode_counter=episode_counter,
                episodes_per_genome=config.episodes_per_genome,
            )
            fitness = _evaluate_genome(
                genome,
                env_factory,
                obs_transform,
                action_transform,
                config,
                seeds,
            )
            result_queue.put((genome_id, fitness))
    except Exception:  # pragma: no cover - propagated to parent
        result_queue.put(("__error__", worker_id))
        raise


class ParallelEvaluator:
    """Multiprocessing evaluator for concurrent genome scoring."""

    def __init__(
        self,
        env_factory: EnvFactory,
        obs_transform: ObsTransform,
        action_transform: ActionTransform,
        *,
        workers: int,
        batch_size: int,
        timeout_s: float | None = None,
        config: EvaluationConfig | None = None,
    ) -> None:
        if workers <= 0:
            msg = "workers must be positive."
            raise ValueError(msg)
        if batch_size <= 0:
            msg = "batch_size must be positive."
            raise ValueError(msg)
        if timeout_s is not None and timeout_s <= 0.0:
            msg = "timeout_s must be positive when provided."
            raise ValueError(msg)

        self.env_factory = env_factory
        self.obs_transform = obs_transform
        self.action_transform = action_transform
        self.workers = workers
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self.config = config or EvaluationConfig()

    def __call__(self, genomes: Mapping[int, Genome], rng: Random) -> dict[int, float]:
        if not genomes:
            return {}

        base_seed = (
            self.config.seed if self.config.seed is not None else rng.getrandbits(32)
        )

        ctx = multiprocessing.get_context("spawn")
        task_queue: multiprocessing.queues.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.queues.Queue[Any] = ctx.Queue()

        processes = [
            ctx.Process(
                target=_worker_loop,
                args=(
                    worker_id,
                    self.env_factory,
                    self.obs_transform,
                    self.action_transform,
                    self.config,
                    base_seed,
                    task_queue,
                    result_queue,
                ),
            )
            for worker_id in range(self.workers)
        ]

        for proc in processes:
            proc.start()

        success = False
        try:
            items = sorted(genomes.items(), key=lambda item: item[0])
            for genome_id, genome in items:
                task_queue.put((genome_id, genome))

            for _ in processes:
                task_queue.put(None)

            results: dict[int, float] = {}
            remaining = len(items)
            while remaining:
                if self.timeout_s is None:
                    genome_id, fitness = result_queue.get()
                else:
                    genome_id, fitness = result_queue.get(timeout=self.timeout_s)
                if genome_id == "__error__":
                    raise RuntimeError(f"Worker {fitness} failed during evaluation.")
                results[genome_id] = fitness
                remaining -= 1

            success = True
            return results
        finally:
            for proc in processes:
                proc.join()
                if success and proc.exitcode not in (0, None):
                    raise RuntimeError(
                        f"Worker process exited with code {proc.exitcode}"
                    )


__all__ = [
    "EvaluationConfig",
    "ParallelEvaluator",
    "SyncEvaluator",
]
