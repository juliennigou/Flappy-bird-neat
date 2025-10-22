"""Headless Flappy Bird environment suitable for evaluator usage."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any


@dataclass(slots=True)
class FlappyEnvConfig:
    """Physics and reward configuration for the Flappy environment."""

    gravity: float = 0.35
    flap_impulse: float = -6.5
    pipe_gap: float = 110.0
    pipe_spacing_px: float = 200.0
    pipe_speed: float = -3.0
    bird_x: float = 56.0
    screen_width: float = 288.0
    screen_height: float = 512.0
    pipe_width: float = 52.0
    alive_reward: float = -0.01
    pipe_passed_reward: float = 1.0
    death_reward: float = 0.0


class FlappyEnv:
    """Minimal deterministic Flappy Bird simulation without rendering."""

    def __init__(self, config: FlappyEnvConfig | None = None) -> None:
        self.config = config or FlappyEnvConfig()
        self._rng = Random()
        self._pipes: list[dict[str, float | bool]] = []
        self._bird_y = 0.0
        self._bird_velocity = 0.0
        self._passed_pipes = 0

    def reset(self, seed: int | None = None) -> dict[str, float]:
        if seed is not None:
            self._rng.seed(seed)
        self._pipes.clear()
        self._spawn_pipe(initial=True)
        self._bird_y = self.config.screen_height / 2.0
        self._bird_velocity = 0.0
        self._passed_pipes = 0
        return self._observation()

    def step(
        self,
        action: float | int | bool,
    ) -> tuple[dict[str, float], float, bool, dict[str, Any]]:
        if bool(action):
            self._bird_velocity = self.config.flap_impulse

        self._bird_velocity += self.config.gravity
        self._bird_y += self._bird_velocity

        reward = self.config.alive_reward

        for pipe in self._pipes:
            pipe["x"] = float(pipe["x"]) + self.config.pipe_speed
            if not pipe["passed"] and pipe["x"] + self.config.pipe_width / 2.0 < self.config.bird_x:
                pipe["passed"] = True
                reward += self.config.pipe_passed_reward
                self._passed_pipes += 1

        if self._pipes and self._pipes[0]["x"] + self.config.pipe_width < 0:
            self._pipes.pop(0)

        if not self._pipes or (
            self.config.screen_width - self._pipes[-1]["x"]
            > self.config.pipe_spacing_px
        ):
            self._spawn_pipe()

        done = self._check_collisions()
        if done:
            reward = self.config.death_reward

        observation = self._observation()
        info = {"pipes_passed": self._passed_pipes}
        return observation, reward, done, info

    def close(self) -> None:
        """No-op for compatibility with gym-style APIs."""

    def _spawn_pipe(self, *, initial: bool = False) -> None:
        min_gap = self.config.pipe_gap / 2.0
        max_gap = self.config.screen_height - min_gap
        gap_center = self._rng.uniform(min_gap, max_gap)
        x_position = (
            self.config.screen_width + self.config.pipe_width
            if initial
            else self.config.screen_width
        )
        self._pipes.append(
            {
                "x": x_position,
                "gap_center": gap_center,
                "passed": False,
            }
        )

    def pipes(self) -> list[dict[str, float | bool]]:
        """Expose current pipe data for rendering helpers."""
        return list(self._pipes)

    def _observation(self) -> dict[str, float]:
        pipe = self._pipes[0]
        return {
            "bird_y": self._bird_y,
            "bird_velocity": self._bird_velocity,
            "pipe_dx": float(pipe["x"]) - self.config.bird_x,
            "pipe_gap_center": float(pipe["gap_center"]),
        }

    def _check_collisions(self) -> bool:
        if self._bird_y <= 0 or self._bird_y >= self.config.screen_height:
            return True

        pipe = self._pipes[0]
        dx = float(pipe["x"]) - self.config.bird_x
        half_width = self.config.pipe_width / 2.0
        if -half_width <= dx <= half_width:
            gap_center = float(pipe["gap_center"])
            half_gap = self.config.pipe_gap / 2.0
            if not (gap_center - half_gap <= self._bird_y <= gap_center + half_gap):
                return True
        return False


__all__ = ["FlappyEnv", "FlappyEnvConfig"]
