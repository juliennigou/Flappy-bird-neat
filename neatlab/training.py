"""Training orchestration utilities for NEAT CLI."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from statistics import mean, median
from time import perf_counter

import yaml

from games.flappy.config import load_env_config
from games.flappy.env.adapter import action_transform, obs_transform
from games.flappy.env.env_core import FlappyEnv, FlappyEnvConfig

from .config import NEATConfig, RunConfig
from .evaluator import EvaluationStats, ParallelEvaluator, SyncEvaluator
from .genes import ConnectionGene, NodeGene, NodeType
from .genome import Genome
from .innovations import InnovationTracker
from .metrics import MetricsRow, MetricsWriter
from .persistence import TrainingCheckpoint, load_checkpoint, save_checkpoint
from .population import MutationOperators, PopulationState
from .reporters import EventLogger
from .species import SpeciesManager


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Resolved file locations used for a training run."""

    root: Path
    metrics: Path
    events: Path
    checkpoint: Path
    champion: Path
    config: Path


def _create_initial_genomes(
    pop_size: int,
    rng: Random,
    tracker: InnovationTracker,
) -> dict[int, Genome]:
    nodes = {
        0: NodeGene(0, NodeType.INPUT, "identity"),
        1: NodeGene(1, NodeType.INPUT, "identity"),
        2: NodeGene(2, NodeType.INPUT, "identity"),
        3: NodeGene(3, NodeType.INPUT, "identity"),
        4: NodeGene(4, NodeType.BIAS, "identity"),
        5: NodeGene(5, NodeType.OUTPUT, "sigmoid"),
    }
    output_id = 5
    innovation_map = {
        node_id: tracker.register(node_id, output_id) for node_id in (0, 1, 2, 3, 4)
    }

    genomes: dict[int, Genome] = {}
    for genome_id in range(pop_size):
        connections: dict[int, ConnectionGene] = {}
        for node_id, innovation in innovation_map.items():
            weight = rng.uniform(-1.0, 1.0)
            connections[innovation] = ConnectionGene(
                innovation=innovation,
                in_node_id=node_id,
                out_node_id=output_id,
                weight=weight,
                enabled=True,
            )
        genomes[genome_id] = Genome(nodes=nodes.copy(), connections=connections)
    return genomes


def _allocate_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = output_root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _build_artifacts(run_dir: Path) -> RunArtifacts:
    return RunArtifacts(
        root=run_dir,
        metrics=run_dir / "metrics.csv",
        events=run_dir / "events.log",
        checkpoint=run_dir / "neat_state.pkl",
        champion=run_dir / "champion.pkl",
        config=run_dir / "config.yml",
    )


def _run_snapshot(run_config: RunConfig) -> dict[str, object]:
    return {
        "neat_config": str(run_config.neat_config),
        "env_config": str(run_config.env_config),
        "headless": run_config.headless,
        "workers": run_config.workers,
        "batch_size": run_config.batch_size,
        "timeout_s": run_config.timeout_s,
        "resume": str(run_config.resume) if run_config.resume else None,
        "save_every": run_config.save_every,
        "output_dir": str(run_config.output_dir),
    }


def _write_config_snapshot(
    artifacts: RunArtifacts,
    run_config: RunConfig,
    neat_config: NEATConfig,
    env_config: FlappyEnvConfig,
) -> None:
    if artifacts.config.exists():
        return
    snapshot = {
        "run": _run_snapshot(run_config),
        "neat": asdict(neat_config),
        "env": asdict(env_config),
    }
    with artifacts.config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, sort_keys=True)


def _create_initial_state(
    neat_config: NEATConfig,
    tracker: InnovationTracker,
) -> PopulationState:
    seed_rng = Random(neat_config.seed)
    species_manager = SpeciesManager(neat_config.species_config())
    genomes = _create_initial_genomes(
        neat_config.population_size,
        seed_rng,
        tracker,
    )
    return PopulationState(
        generation=0,
        genomes=genomes,
        rng=Random(seed_rng.getrandbits(32)),
        species_manager=species_manager,
        config=neat_config.population_config(),
        reproduction_config=neat_config.reproduction_config(),
    )


def _create_evaluator(
    run_config: RunConfig,
    env_config: FlappyEnvConfig,
    evaluation_config,
) -> SyncEvaluator | ParallelEvaluator:
    def env_factory() -> FlappyEnv:
        return FlappyEnv(env_config)

    if run_config.workers > 1:
        return ParallelEvaluator(
            env_factory,
            obs_transform,
            action_transform,
            workers=run_config.workers,
            batch_size=run_config.batch_size,
            timeout_s=run_config.timeout_s,
            config=evaluation_config,
        )
    return SyncEvaluator(
        env_factory,
        obs_transform,
        action_transform,
        config=evaluation_config,
    )


def _normalise_checkpoint_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "neat_state.pkl"
    if not candidate.exists():
        msg = f"Checkpoint file not found: {candidate}"
        raise FileNotFoundError(msg)
    return candidate


def _initialise_run(
    run_config: RunConfig,
    neat_config: NEATConfig,
    env_config: FlappyEnvConfig,
) -> tuple[RunArtifacts, PopulationState, Genome | None, float, InnovationTracker]:
    if run_config.resume:
        checkpoint_path = _normalise_checkpoint_path(run_config.resume)
        checkpoint = load_checkpoint(checkpoint_path)
        population_state = checkpoint.population_state
        best_genome = checkpoint.best_genome.copy()
        best_fitness = checkpoint.best_fitness
        run_dir = checkpoint_path.parent
        tracker = InnovationTracker.from_snapshot(checkpoint.innovation_snapshot)
    else:
        run_dir = _allocate_run_dir(run_config.output_dir)
        tracker = InnovationTracker()
        population_state = _create_initial_state(neat_config, tracker)
        best_genome = None
        best_fitness = float("-inf")

    artifacts = _build_artifacts(run_dir)
    _write_config_snapshot(artifacts, run_config, neat_config, env_config)
    return artifacts, population_state, best_genome, best_fitness, tracker


def _save_champion(
    artifacts: RunArtifacts,
    generation: int,
    fitness: float,
    genome: Genome,
) -> None:
    payload = {
        "generation": generation,
        "fitness": fitness,
        "genome": genome.copy(),
    }
    with artifacts.champion.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _persist_state(
    artifacts: RunArtifacts,
    population_state: PopulationState,
    best_genome: Genome | None,
    best_fitness: float,
    tracker: InnovationTracker,
) -> None:
    if not population_state.genomes:
        msg = "Cannot persist checkpoint with empty population."
        raise ValueError(msg)
    genome = (
        best_genome.copy()
        if best_genome is not None
        else next(iter(population_state.genomes.values())).copy()
    )
    checkpoint = TrainingCheckpoint(
        generation=population_state.generation,
        population_state=population_state,
        best_genome=genome,
        best_fitness=best_fitness,
        innovation_snapshot=tracker.to_snapshot(),
    )
    save_checkpoint(artifacts.checkpoint, checkpoint)


def _should_save(generation: int, interval: int | None) -> bool:
    if interval is None or interval <= 0:
        return False
    return generation % interval == 0


def _build_mutation_operators(
    neat_config: NEATConfig,
    tracker: InnovationTracker,
) -> MutationOperators:
    return MutationOperators(
        tracker=tracker,
        weight=neat_config.weight_mutation_config(),
        add_connection=neat_config.add_connection_config(),
        add_node=neat_config.add_node_config(),
        add_connection_rate=neat_config.add_connection_rate,
        add_node_rate=neat_config.add_node_rate,
        crossover_rate=neat_config.crossover_rate,
        crossover=neat_config.crossover_config(),
    )


def run_training(run_config: RunConfig, neat_config: NEATConfig) -> None:
    env_config = load_env_config(run_config.env_config)
    evaluation_config = neat_config.evaluation_config()

    (
        artifacts,
        population_state,
        best_genome,
        best_fitness,
        tracker,
    ) = _initialise_run(
        run_config,
        neat_config,
        env_config,
    )

    evaluator = _create_evaluator(run_config, env_config, evaluation_config)
    operators = _build_mutation_operators(neat_config, tracker)

    if best_genome is not None and not artifacts.champion.exists():
        _save_champion(
            artifacts,
            population_state.generation,
            best_fitness,
            best_genome,
        )

    if run_config.resume:
        print(f"[train] resuming from checkpoint: {artifacts.checkpoint}")
    else:
        print(f"[train] run directory: {artifacts.root}")

    with MetricsWriter(artifacts.metrics) as metrics_writer, EventLogger(
        artifacts.events
    ) as logger:
        mode = "resumed" if run_config.resume else "started"
        logger.log(f"Training {mode} at {artifacts.root}")
        logger.log(
            f"Configs -> neat={run_config.neat_config} env={run_config.env_config}"
        )

        while population_state.generation < neat_config.max_generations:
            generation = population_state.generation

            start_time = perf_counter()
            fitnesses = population_state.evaluate(evaluator)
            eval_time = perf_counter() - start_time

            best_id = max(fitnesses, key=fitnesses.__getitem__)
            generation_best = fitnesses[best_id]

            if generation_best > best_fitness or best_genome is None:
                best_fitness = generation_best
                best_genome = population_state.genomes[best_id].copy()
                _save_champion(artifacts, generation, best_fitness, best_genome)
                logger.log(
                    f"New champion at generation {generation} (fitness={best_fitness:.3f})."
                )

            stats: EvaluationStats = evaluator.last_stats
            steps_per_sec = (
                stats.steps / eval_time if eval_time > 0.0 and stats.steps > 0 else 0.0
            )

            species = population_state.speciate()
            species_count = len(species)
            fitness_values = list(fitnesses.values())
            mean_fitness = mean(fitness_values)
            median_fitness = median(fitness_values)

            metrics_writer.append(
                MetricsRow(
                    generation=generation,
                    population_size=len(population_state.genomes),
                    species_count=species_count,
                    best_fitness=generation_best,
                    mean_fitness=mean_fitness,
                    median_fitness=median_fitness,
                    eval_time_s=eval_time,
                    steps_per_sec=steps_per_sec,
                )
            )

            logger.log(
                "Generation "
                f"{generation}: best={generation_best:.3f} "
                f"mean={mean_fitness:.3f} median={median_fitness:.3f} "
                f"species={species_count} steps={stats.steps}"
            )
            print(f"Generation {generation}: best fitness {generation_best:.2f}")

            if generation_best >= neat_config.fitness_threshold:
                logger.log("Fitness threshold reached; stopping.")
                print("Fitness threshold reached, stopping training.")
                break

            population_state.reproduce(species, operators)

            if _should_save(population_state.generation, run_config.save_every):
                _persist_state(
                    artifacts,
                    population_state,
                    best_genome,
                    best_fitness,
                    tracker,
                )
                logger.log(
                    f"Checkpoint saved at generation {population_state.generation}."
                )

        _persist_state(
            artifacts,
            population_state,
            best_genome,
            best_fitness,
            tracker,
        )
        logger.log("Final checkpoint saved.")


__all__ = ["run_training"]
