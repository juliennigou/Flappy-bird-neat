"""Configuration helpers for the Flappy environment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from .env.env_core import FlappyEnvConfig


def load_env_config(path: Path) -> FlappyEnvConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        msg = f"Expected mapping in YAML file: {path}"
        raise ValueError(msg)
    return FlappyEnvConfig(
        gravity=float(data.get("gravity", 0.35)),
        flap_impulse=float(data.get("flap_impulse", -6.5)),
        pipe_gap=float(data.get("pipe_gap", 110.0)),
        pipe_spacing_px=float(data.get("pipe_spacing_px", 200.0)),
        pipe_speed=float(data.get("pipe_speed", -3.0)),
        bird_x=float(data.get("bird_x", 56.0)),
        screen_width=float(data.get("screen_width", 288.0)),
        screen_height=float(data.get("screen_height", 512.0)),
        pipe_width=float(data.get("pipe_width", 52.0)),
        alive_reward=float(data.get("reward", {}).get("alive_step", -0.01)),
        pipe_passed_reward=float(data.get("reward", {}).get("pipe_passed", 1.0)),
        death_reward=float(data.get("reward", {}).get("death", 0.0)),
    )


__all__ = ["load_env_config"]
