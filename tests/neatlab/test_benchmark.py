from __future__ import annotations

from pathlib import Path

from games.flappy.config import load_env_config
from neatlab.benchmark import run_benchmark


def _env_yaml_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "games" / "flappy" / "configs" / "env.yml"


def test_run_benchmark_returns_results() -> None:
    env_config = load_env_config(_env_yaml_path())
    results = run_benchmark(
        env_config,
        steps_target=20,
        population_size=2,
        episodes_per_genome=1,
        worker_counts=[1],
        batch_size=1,
        seed=42,
    )

    assert len(results) == 1
    result = results[0]
    assert result.workers == 1
    assert result.steps > 0
    assert result.steps_per_sec >= 0.0
    assert result.iterations >= 1
