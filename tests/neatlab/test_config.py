from __future__ import annotations

from pathlib import Path

from neatlab.config import NEATConfig, RunConfig, load_neat_config, load_run_config


def test_load_neat_config(tmp_path: Path) -> None:
    neat_yaml = tmp_path / "neat.yml"
    neat_yaml.write_text(
        """
population_size: 20
elitism: 2
survival_threshold: 0.3
max_stagnation: 5
episodes_per_genome: 3
max_generations: 100
fitness_threshold: 25.0
seed: 99
c1: 1.5
""",
        encoding="utf-8",
    )
    config = load_neat_config(neat_yaml)
    assert isinstance(config, NEATConfig)
    assert config.population_size == 20
    assert config.seed == 99
    assert config.c1 == 1.5


def test_load_run_config(tmp_path: Path) -> None:
    neat_yaml = tmp_path / "neat.yml"
    neat_yaml.write_text(
        "population_size: 5\n"
        "elitism: 1\n"
        "survival_threshold: 0.5\n"
        "max_stagnation: 2\n"
        "episodes_per_genome: 1\n"
        "max_generations: 5\n"
        "fitness_threshold: 10\n",
        encoding="utf-8",
    )

    env_yaml = tmp_path / "env.yml"
    env_yaml.write_text("pipe_gap: 120\n", encoding="utf-8")

    run_yaml = tmp_path / "run.yml"
    run_yaml.write_text(
        """
neat_config: neat.yml
env_config: env.yml
headless: true
workers: 2
batch_size: 4
timeout_s: 5
save_every: 1
""",
        encoding="utf-8",
    )

    run_config = load_run_config(run_yaml)
    assert isinstance(run_config, RunConfig)
    assert run_config.neat_config.name == "neat.yml"
    assert run_config.workers == 2
    assert run_config.timeout_s == 5.0
