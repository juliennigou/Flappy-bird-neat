"""Training orchestration utilities for NEAT CLI."""

from __future__ import annotations

from random import Random

from games.flappy.config import load_env_config
from games.flappy.env.adapter import action_transform, obs_transform
from games.flappy.env.env_core import FlappyEnv

from .config import NEATConfig, RunConfig
from .evaluator import ParallelEvaluator, SyncEvaluator
from .genes import ConnectionGene, NodeGene, NodeType
from .genome import Genome, WeightMutationConfig
from .population import PopulationState
from .species import SpeciesManager


def _create_initial_genomes(pop_size: int, rng: Random) -> dict[int, Genome]:
    nodes = {
        0: NodeGene(0, NodeType.INPUT, "identity"),
        1: NodeGene(1, NodeType.INPUT, "identity"),
        2: NodeGene(2, NodeType.INPUT, "identity"),
        3: NodeGene(3, NodeType.INPUT, "identity"),
        4: NodeGene(4, NodeType.BIAS, "identity"),
        5: NodeGene(5, NodeType.OUTPUT, "sigmoid"),
    }
    genomes: dict[int, Genome] = {}
    for genome_id in range(pop_size):
        connections = {}
        innovation = 0
        for node_id in (0, 1, 2, 3, 4):
            weight = rng.uniform(-1.0, 1.0)
            connections[innovation] = ConnectionGene(
                innovation=innovation,
                in_node_id=node_id,
                out_node_id=5,
                weight=weight,
                enabled=True,
            )
            innovation += 1
        genomes[genome_id] = Genome(nodes=nodes.copy(), connections=connections)
    return genomes


def _mutate_genome(genome: Genome, rng: Random) -> None:
    config = WeightMutationConfig(mutate_rate=0.9, perturb_sd=0.6, reset_rate=0.1)
    genome.mutate_weight(rng, config)


def run_training(run_config: RunConfig, neat_config: NEATConfig) -> None:
    env_config = load_env_config(run_config.env_config)
    rng = Random(neat_config.seed)

    species_manager = SpeciesManager(neat_config.species_config())
    population_state = PopulationState(
        generation=0,
        genomes=_create_initial_genomes(neat_config.population_size, rng),
        rng=Random(rng.getrandbits(32)),
        species_manager=species_manager,
        config=neat_config.population_config(),
        reproduction_config=neat_config.reproduction_config(),
    )

    def env_factory() -> FlappyEnv:
        return FlappyEnv(env_config)
    evaluation_config = neat_config.evaluation_config()

    evaluator: SyncEvaluator | ParallelEvaluator
    if run_config.workers > 1:
        evaluator = ParallelEvaluator(
            env_factory,
            obs_transform,
            action_transform,
            workers=run_config.workers,
            batch_size=run_config.batch_size,
            timeout_s=run_config.timeout_s,
            config=evaluation_config,
        )
    else:
        evaluator = SyncEvaluator(
            env_factory,
            obs_transform,
            action_transform,
            config=evaluation_config,
        )

    for generation in range(neat_config.max_generations):
        fitnesses = population_state.evaluate(evaluator)
        best_id = max(fitnesses, key=fitnesses.__getitem__)
        best_fitness = fitnesses[best_id]
        print(f"Generation {generation}: best fitness {best_fitness:.2f}")

        if best_fitness >= neat_config.fitness_threshold:
            print("Fitness threshold reached, stopping training.")
            break

        species = population_state.speciate()
        population_state.reproduce(species, _mutate_genome)


__all__ = ["run_training"]
