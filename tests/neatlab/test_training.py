from __future__ import annotations

import csv
import pickle
from pathlib import Path

from neatlab.config import NEATConfig, RunConfig
from neatlab.persistence import load_checkpoint
from neatlab.training import run_training


def _env_yaml_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "games" / "flappy" / "configs" / "env.yml"


def _basic_neat_config(max_generations: int) -> NEATConfig:
    return NEATConfig(
        population_size=4,
        elitism=1,
        survival_threshold=0.5,
        max_stagnation=1,
        episodes_per_genome=1,
        max_generations=max_generations,
        fitness_threshold=9999.0,
        seed=123,
    )


def _run_config(output_dir: Path, resume: Path | None = None) -> RunConfig:
    return RunConfig(
        neat_config=output_dir / "neat.yml",
        env_config=_env_yaml_path(),
        headless=True,
        workers=1,
        batch_size=1,
        timeout_s=None,
        resume=resume,
        save_every=1,
        output_dir=output_dir,
    )


def _list_run_dirs(output_dir: Path) -> list[Path]:
    return sorted(path for path in output_dir.iterdir() if path.is_dir())


def test_run_training_writes_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "runs"
    run_training(_run_config(output_dir), _basic_neat_config(max_generations=2))

    run_dirs = _list_run_dirs(output_dir)
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    metrics_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "neat_state.pkl"
    champion_path = run_dir / "champion.pkl"
    log_path = run_dir / "events.log"
    config_path = run_dir / "config.yml"

    for path in (metrics_path, checkpoint_path, champion_path, log_path, config_path):
        assert path.exists(), f"Missing artifact: {path}"

    with metrics_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["generation"] == "0"

    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint.generation == 2
    assert checkpoint.best_genome is not None
    assert checkpoint.innovation_snapshot.next_innovation >= 5

    with champion_path.open("rb") as handle:
        champion_payload = pickle.load(handle)
    assert champion_payload["generation"] in (0, 1)
    assert "genome" in champion_payload


def test_run_training_resume_appends_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "runs"
    initial_config = _basic_neat_config(max_generations=2)
    run_training(_run_config(output_dir), initial_config)

    run_dir = _list_run_dirs(output_dir)[0]
    checkpoint_path = run_dir / "neat_state.pkl"

    with (run_dir / "metrics.csv").open("r", encoding="utf-8") as handle:
        initial_rows = len(list(csv.DictReader(handle)))

    resumed_config = _basic_neat_config(max_generations=3)
    run_training(
        _run_config(output_dir, resume=checkpoint_path),
        resumed_config,
    )

    with (run_dir / "metrics.csv").open("r", encoding="utf-8") as handle:
        resumed_rows = len(list(csv.DictReader(handle)))
    assert resumed_rows == resumed_config.max_generations
    assert resumed_rows > initial_rows

    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint.generation == 3
    assert checkpoint.innovation_snapshot.next_innovation >= 5
