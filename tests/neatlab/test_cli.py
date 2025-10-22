from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "neatlab.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "train" in result.stdout


def test_cli_train_dry_run(tmp_path: Path) -> None:
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
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "neatlab.cli",
            "train",
            "--config",
            str(run_yaml),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "configuration validated" in result.stdout
