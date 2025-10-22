"""Utilities for recording training metrics to disk."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, IO


@dataclass(frozen=True, slots=True)
class MetricsRow:
    """Row of aggregate statistics produced for each generation."""

    generation: int
    population_size: int
    species_count: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    eval_time_s: float
    steps_per_sec: float


class MetricsWriter:
    """CSV-backed writer that appends metrics rows incrementally."""

    _fieldnames = [
        "generation",
        "population_size",
        "species_count",
        "best_fitness",
        "mean_fitness",
        "median_fitness",
        "eval_time_s",
        "steps_per_sec",
    ]

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        exists = self._path.exists()
        mode = "a" if exists else "w"
        self._handle: IO[str] = self._path.open(mode, encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self._fieldnames)
        if not exists:
            self._writer.writeheader()
            self._handle.flush()

    def __enter__(self) -> MetricsWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    def append(self, row: MetricsRow) -> None:
        """Append a metrics row and flush to disk."""
        self._writer.writerow(asdict(row))
        self._handle.flush()

    def close(self) -> None:
        """Release the underlying file handle if still open."""
        if not self._handle.closed:
            self._handle.close()

    @property
    def path(self) -> Path:
        """Return the destination path for the CSV file."""
        return self._path


__all__ = ["MetricsRow", "MetricsWriter"]
