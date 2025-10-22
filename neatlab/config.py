"""Configuration loading utilities for NEAT runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .evaluator import EvaluationConfig
from .population import PopulationConfig
from .reproduction import ReproductionConfig
from .species import SpeciesConfig


@dataclass(slots=True)
class NEATConfig:
    population_size: int
    elitism: int
    survival_threshold: float
    max_stagnation: int
    episodes_per_genome: int
    max_generations: int
    fitness_threshold: float
    seed: int | None = None
    c1: float = 1.0
    c2: float = 1.0
    c3: float = 0.4
    compatibility_threshold: float = 3.0
    target_species: int = 5
    species_adjust_period: int = 1
    adjust_rate: float = 0.1
    min_compatibility_threshold: float = 0.5
    max_compatibility_threshold: float = 5.0

    def population_config(self) -> PopulationConfig:
        return PopulationConfig(
            population_size=self.population_size,
            elitism=self.elitism,
            survival_threshold=self.survival_threshold,
            max_stagnation=self.max_stagnation,
        )

    def species_config(self) -> SpeciesConfig:
        return SpeciesConfig(
            c1=self.c1,
            c2=self.c2,
            c3=self.c3,
            compatibility_threshold=self.compatibility_threshold,
            target_species=self.target_species,
            species_adjust_period=self.species_adjust_period,
            adjust_rate=self.adjust_rate,
            min_compatibility_threshold=self.min_compatibility_threshold,
            max_compatibility_threshold=self.max_compatibility_threshold,
        )

    def reproduction_config(self) -> ReproductionConfig:
        return ReproductionConfig(
            elitism=self.elitism,
            survival_threshold=self.survival_threshold,
        )

    def evaluation_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            episodes_per_genome=self.episodes_per_genome,
            seed=self.seed,
        )


@dataclass(slots=True)
class RunConfig:
    neat_config: Path
    env_config: Path
    headless: bool = True
    workers: int = 1
    batch_size: int = 1
    timeout_s: float | None = None
    resume: Path | None = None
    save_every: int | None = None

    def resolve(self, base_path: Path) -> RunConfig:
        return RunConfig(
            neat_config=(base_path / self.neat_config).resolve(),
            env_config=(base_path / self.env_config).resolve(),
            headless=self.headless,
            workers=self.workers,
            batch_size=self.batch_size,
            timeout_s=self.timeout_s,
            resume=(base_path / self.resume).resolve() if self.resume else None,
            save_every=self.save_every,
        )


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        msg = f"Expected mapping in YAML file: {path}"
        raise ValueError(msg)
    return data


def load_neat_config(path: Path) -> NEATConfig:
    data = _load_yaml(path)
    return NEATConfig(
        population_size=int(data.get("population_size", data.get("pop_size", 10))),
        elitism=int(data.get("elitism", 1)),
        survival_threshold=float(data.get("survival_threshold", 0.5)),
        max_stagnation=int(data.get("max_stagnation", 10)),
        episodes_per_genome=int(data.get("episodes_per_genome", 1)),
        max_generations=int(data.get("max_generations", 50)),
        fitness_threshold=float(data.get("fitness_threshold", 10.0)),
        seed=(int(data["seed"]) if "seed" in data else None),
        c1=float(data.get("c1", 1.0)),
        c2=float(data.get("c2", 1.0)),
        c3=float(data.get("c3", 0.4)),
        compatibility_threshold=float(data.get("compatibility_threshold", 3.0)),
        target_species=int(data.get("target_species", 5)),
        species_adjust_period=int(data.get("species_adjust_period", 1)),
        adjust_rate=float(data.get("adjust_rate", 0.1)),
        min_compatibility_threshold=float(data.get("min_compatibility_threshold", 0.5)),
        max_compatibility_threshold=float(data.get("max_compatibility_threshold", 5.0)),
    )


def load_run_config(path: Path) -> RunConfig:
    data = _load_yaml(path)
    neat_path = data.get("neat_config")
    env_path = data.get("env_config")
    if neat_path is None or env_path is None:
        msg = "run.yml must specify 'neat_config' and 'env_config' paths"
        raise ValueError(msg)
    base = path.parent
    run = RunConfig(
        neat_config=Path(neat_path),
        env_config=Path(env_path),
        headless=bool(data.get("headless", True)),
        workers=int(data.get("workers", 1)),
        batch_size=int(data.get("batch_size", 1)),
        timeout_s=(
            float(data["timeout_s"])
            if data.get("timeout_s") is not None
            else None
        ),
        resume=(Path(data["resume"]) if data.get("resume") else None),
        save_every=(int(data["save_every"]) if data.get("save_every") else None),
    )
    return run.resolve(base)


__all__ = ["NEATConfig", "RunConfig", "load_neat_config", "load_run_config"]
