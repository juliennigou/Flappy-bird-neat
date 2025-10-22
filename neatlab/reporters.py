"""Simple logging/reporting helpers for long-running training sessions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class EventLogger:
    """Append-only text logger with ISO timestamps."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")

    def __enter__(self) -> EventLogger:
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

    def log(self, message: str) -> None:
        """Append a timestamped message to the log."""
        timestamp = datetime.now(timezone.utc).isoformat()
        self._handle.write(f"{timestamp} {message}\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        if not self._handle.closed:
            self._handle.close()

    @property
    def path(self) -> Path:
        """Return the backing log path."""
        return self._path


__all__ = ["EventLogger"]
