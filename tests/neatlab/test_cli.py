from __future__ import annotations

from pathlib import Path

from neatlab import cli


def test_cli_help() -> None:
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert "train" in help_text


def _write_neat_config(path: Path, *, fitness_threshold: float) -> None:
    path.write_text(
        "population_size: 2\n"
        "elitism: 1\n"
        "survival_threshold: 0.5\n"
        "max_stagnation: 2\n"
        "episodes_per_genome: 1\n"
        "max_generations: 1\n"
        f"fitness_threshold: {fitness_threshold}\n",
        encoding="utf-8",
    )


def _write_env_config(path: Path) -> None:
    path.write_text("pipe_gap: 120\n", encoding="utf-8")


def _write_run_config(path: Path) -> None:
    path.write_text(
        """
neat_config: neat.yml
env_config: env.yml
headless: true
workers: 1
batch_size: 1
""",
        encoding="utf-8",
    )


def test_cli_train_dry_run(tmp_path: Path) -> None:
    neat_yaml = tmp_path / "neat.yml"
    _write_neat_config(neat_yaml, fitness_threshold=10.0)
    env_yaml = tmp_path / "env.yml"
    _write_env_config(env_yaml)
    run_yaml = tmp_path / "run.yml"
    _write_run_config(run_yaml)

    code = cli.main(["train", "--config", str(run_yaml), "--dry-run"])
    assert code == 0


def test_cli_train_executes(tmp_path: Path) -> None:
    neat_yaml = tmp_path / "neat.yml"
    _write_neat_config(neat_yaml, fitness_threshold=-1.0)
    env_yaml = tmp_path / "env.yml"
    _write_env_config(env_yaml)
    run_yaml = tmp_path / "run.yml"
    _write_run_config(run_yaml)

    code = cli.main(["train", "--config", str(run_yaml)])
    assert code == 0


def test_cli_play_dry_run(tmp_path: Path) -> None:
    env_yaml = tmp_path / "env.yml"
    _write_env_config(env_yaml)

    code = cli.main(["play", "--env-config", str(env_yaml), "--dry-run"])
    assert code == 0
