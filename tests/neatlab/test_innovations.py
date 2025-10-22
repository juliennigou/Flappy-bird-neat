from __future__ import annotations

import pytest
from neatlab.innovations import InnovationSnapshot, InnovationTracker


def test_register_assigns_monotonic_ids() -> None:
    tracker = InnovationTracker()
    innovation_a = tracker.register(1, 2)
    innovation_b = tracker.register(1, 3)

    assert innovation_a == 0
    assert innovation_b == innovation_a + 1
    # Re-registering the same connection reuses the original id.
    assert tracker.register(1, 2) == innovation_a


def test_tracker_peek_and_contains() -> None:
    tracker = InnovationTracker()
    tracker.register(1, 2)

    assert tracker.peek(1, 2) == 0
    assert (1, 2) in tracker
    assert tracker.peek(2, 3) is None
    assert (2, 3) not in tracker


def test_snapshot_roundtrip_preserves_state() -> None:
    tracker = InnovationTracker()
    original_ids = [
        tracker.register(0, 1),
        tracker.register(1, 2),
        tracker.register(2, 3),
    ]
    snapshot = tracker.to_snapshot()

    restored = InnovationTracker.from_snapshot(snapshot)
    assert restored.peek(0, 1) == original_ids[0]
    assert restored.peek(1, 2) == original_ids[1]
    assert restored.peek(2, 3) == original_ids[2]
    assert restored.next_innovation == tracker.next_innovation


def test_from_snapshot_accepts_plain_mapping() -> None:
    snapshot = {"next_innovation": 4, "pairs": [(0, 1, 1), (1, 2, 3)]}
    tracker = InnovationTracker.from_snapshot(snapshot)

    assert tracker.peek(0, 1) == 1
    assert tracker.peek(1, 2) == 3
    assert tracker.next_innovation == 4


def test_snapshot_to_mapping_and_items_view() -> None:
    tracker = InnovationTracker()
    tracker.register(2, 3)
    tracker.register(3, 4)

    snapshot = tracker.to_snapshot()
    mapping = snapshot.to_mapping()
    assert mapping[(2, 3)] == 0
    assert mapping[(3, 4)] == 1
    assert len(tracker) == 2
    assert dict(tracker.items()) == {(2, 3): 0, (3, 4): 1}


def test_negative_inputs_raise_value_error() -> None:
    tracker = InnovationTracker()

    with pytest.raises(ValueError):
        tracker.register(-1, 0)

    with pytest.raises(ValueError):
        tracker.register(0, -5)

    snapshot = InnovationSnapshot(next_innovation=1, pairs=((0, 1, -1),))
    with pytest.raises(ValueError):
        InnovationTracker.from_snapshot(snapshot)


def test_tracker_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        InnovationTracker(next_innovation=-5)


def test_from_snapshot_missing_keys() -> None:
    with pytest.raises(ValueError):
        InnovationTracker.from_snapshot({"pairs": []})


def test_from_snapshot_rejects_bad_pair_shape() -> None:
    snapshot = {"next_innovation": 3, "pairs": [(0, 1)]}
    with pytest.raises(ValueError):
        InnovationTracker.from_snapshot(snapshot)


def test_from_snapshot_accepts_string_values() -> None:
    snapshot = {"next_innovation": "5", "pairs": [("1", "2", "3")]}
    tracker = InnovationTracker.from_snapshot(snapshot)

    assert tracker.peek(1, 2) == 3
    assert tracker.next_innovation == 5


def test_from_snapshot_rejects_non_convertible_values() -> None:
    class NoInt:
        pass

    snapshot = {"next_innovation": NoInt(), "pairs": [(0, 1, 0)]}
    with pytest.raises(ValueError):
        InnovationTracker.from_snapshot(snapshot)


def test_from_snapshot_accepts_int_like_objects() -> None:
    class IntLike:
        def __init__(self, value: int) -> None:
            self.value = value

        def __int__(self) -> int:
            return self.value

    snapshot = {
        "next_innovation": IntLike(7),
        "pairs": [(IntLike(2), IntLike(3), IntLike(5))],
    }
    tracker = InnovationTracker.from_snapshot(snapshot)

    assert tracker.peek(2, 3) == 5
    assert tracker.next_innovation == 7


def test_restore_rejects_duplicate_innovations() -> None:
    snapshot = InnovationSnapshot(
        next_innovation=3,
        pairs=((0, 1, 1), (1, 2, 1)),
    )
    with pytest.raises(ValueError):
        InnovationTracker.from_snapshot(snapshot)
