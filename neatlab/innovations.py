"""Innovation tracking utilities for NEAT genome connectivity."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, SupportsInt, cast

InnovationKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class InnovationSnapshot:
    """Serializable snapshot of innovation assignments.

    Attributes:
        next_innovation: The next innovation identifier that will be assigned.
        pairs: Sorted tuple of (in_id, out_id, innovation_id) triples.
    """

    next_innovation: int
    pairs: tuple[tuple[int, int, int], ...]

    def to_mapping(self) -> dict[InnovationKey, int]:
        """Convert snapshot pairs back to a dictionary of innovation mappings."""
        return {(in_id, out_id): innovation for in_id, out_id, innovation in self.pairs}


@dataclass(slots=True)
class InnovationTracker:
    """Assigns and persists unique innovation identifiers for gene connections."""

    next_innovation: int = 0
    _mapping: dict[InnovationKey, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.next_innovation < 0:
            msg = "next_innovation must be non-negative."
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of registered innovation mappings."""
        return len(self._mapping)

    def __contains__(self, key: object) -> bool:
        """Indicate whether a connection pair already has an innovation id."""
        return key in self._mapping

    def register(self, in_node_id: int, out_node_id: int) -> int:
        """Register a connection and return its innovation identifier.

        Args:
            in_node_id: Identifier of the source node.
            out_node_id: Identifier of the destination node.

        Returns:
            The stable innovation identifier for the connection.
        """
        key = self._normalize_key((in_node_id, out_node_id))
        existing = self._mapping.get(key)
        if existing is not None:
            return existing

        innovation = self.next_innovation
        self._mapping[key] = innovation
        self.next_innovation += 1
        return innovation

    def peek(self, in_node_id: int, out_node_id: int) -> int | None:
        """Return the innovation id for a connection if it already exists."""
        key = (in_node_id, out_node_id)
        return self._mapping.get(key)

    def items(self) -> Iterator[tuple[InnovationKey, int]]:
        """Iterate over registered innovation mappings."""
        return iter(self._mapping.items())

    def to_snapshot(self) -> InnovationSnapshot:
        """Produce a snapshot suitable for persistence."""
        pairs = tuple(
            sorted(
                (
                    (in_id, out_id, innovation)
                    for (in_id, out_id), innovation in self._mapping.items()
                ),
                key=lambda triple: triple[2],
            )
        )
        return InnovationSnapshot(next_innovation=self.next_innovation, pairs=pairs)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: InnovationSnapshot | Mapping[str, object],
    ) -> InnovationTracker:
        """Restore a tracker from a snapshot or snapshot-like mapping."""
        if isinstance(snapshot, InnovationSnapshot):
            next_innovation = snapshot.next_innovation
            pairs = snapshot.pairs
        else:
            try:
                next_candidate = snapshot["next_innovation"]
                raw_pairs = snapshot["pairs"]
            except KeyError as error:
                msg = f"Snapshot is missing required key: {error.args[0]}"
                raise ValueError(msg) from error
            next_innovation = cls._coerce_int(next_candidate, label="next_innovation")

            pairs_list: list[tuple[int, int, int]] = []
            for triple in cast(Iterable[Iterable[Any]], raw_pairs):
                converted: list[int] = []
                for part in triple:
                    converted.append(cls._coerce_int(part, label="snapshot pair value"))
                if len(converted) != 3:
                    msg = "Snapshot pairs must contain exactly three elements."
                    raise ValueError(msg)
                pairs_list.append((converted[0], converted[1], converted[2]))
            pairs = tuple(pairs_list)

        tracker = cls(next_innovation=next_innovation)
        mapping = {(in_id, out_id): innovation for in_id, out_id, innovation in pairs}
        tracker._restore(mapping)
        return tracker

    def _restore(self, mapping: Mapping[InnovationKey, int]) -> None:
        """Restore the internal mapping ensuring invariants hold."""
        normalized = {
            self._normalize_key(key): self._validate_innovation(value)
            for key, value in mapping.items()
        }
        if len(normalized) != len(set(normalized.values())):
            msg = "Duplicate innovation identifiers detected in snapshot."
            raise ValueError(msg)

        self._mapping = dict(normalized)
        if self._mapping:
            max_innovation = max(self._mapping.values())
            if max_innovation >= self.next_innovation:
                self.next_innovation = max_innovation + 1

    @staticmethod
    def _normalize_key(key: InnovationKey) -> InnovationKey:
        in_id, out_id = key
        InnovationTracker._ensure_non_negative(in_id, label="in_node_id")
        InnovationTracker._ensure_non_negative(out_id, label="out_node_id")
        return in_id, out_id

    @staticmethod
    def _validate_innovation(value: int) -> int:
        InnovationTracker._ensure_non_negative(value, label="innovation id")
        return value

    @staticmethod
    def _ensure_non_negative(value: int, *, label: str) -> None:
        if value < 0:
            msg = f"{label} must be non-negative."
            raise ValueError(msg)

    @staticmethod
    def _coerce_int(value: object, *, label: str) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, (str, bytes, bytearray)):
            try:
                return int(value)
            except ValueError as error:
                msg = f"{label} must be convertible to int."
                raise ValueError(msg) from error
        if hasattr(value, "__int__"):
            try:
                return int(cast(SupportsInt, value))
            except (TypeError, ValueError) as error:
                msg = f"{label} must be convertible to int."
                raise ValueError(msg) from error
        msg = f"{label} must be convertible to int."
        raise ValueError(msg)


__all__ = ["InnovationTracker", "InnovationSnapshot", "InnovationKey"]
