"""Observation and action transforms for the Flappy Bird environment."""

from __future__ import annotations

from typing import Sequence


def obs_transform(observation: dict[str, float]) -> list[float]:
    return [
        observation["bird_y"],
        observation["bird_velocity"],
        observation["pipe_dx"],
        observation["pipe_gap_center"],
    ]


def action_transform(outputs: Sequence[float]) -> int:
    return 1 if outputs and outputs[0] > 0.5 else 0


__all__ = ["obs_transform", "action_transform"]
