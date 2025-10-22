"""Pygame-based renderer for the Flappy headless environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .env_core import FlappyEnv, FlappyEnvConfig

try:  # pragma: no cover - optional dependency
    import pygame
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pygame = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing references only
    import pygame.surface


@dataclass(slots=True)
class VisualConfig:
    """Rendering configuration for the Flappy visual environment."""

    scale: float = 1.0
    bird_color: tuple[int, int, int] = (255, 255, 0)
    pipe_color: tuple[int, int, int] = (0, 200, 0)
    background_color: tuple[int, int, int] = (24, 24, 24)


class FlappyVisualEnv:
    """Wraps ``FlappyEnv`` and provides Pygame-based rendering."""

    def __init__(
        self,
        env: FlappyEnv | None = None,
        *,
        env_config: FlappyEnvConfig | None = None,
        visual_config: VisualConfig | None = None,
        init_pygame: bool = True,
    ) -> None:
        if pygame is None:  # pragma: no cover - guarded by tests
            msg = "Pygame is required for FlappyVisualEnv."
            raise RuntimeError(msg)

        self.env = env or FlappyEnv(env_config)
        self.visual = visual_config or VisualConfig()
        self.scale = self.visual.scale
        self._screen: pygame.surface.Surface | None = None
        self._last_obs: dict[str, float] | None = None

        width = int(self.env.config.screen_width * self.scale)
        height = int(self.env.config.screen_height * self.scale)
        self._surface = pygame.Surface((width, height))

        if init_pygame:
            pygame.display.init()
            self._screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Flappy Bird (visual)")

    @property
    def surface(self) -> "pygame.surface.Surface":
        return self._surface

    def reset(self, seed: int | None = None) -> dict[str, float]:
        observation = self.env.reset(seed=seed)
        self._last_obs = observation
        return observation

    def step(
        self,
        action: float | int | bool,
    ) -> tuple[dict[str, float], float, bool, dict[str, Any]]:
        observation, reward, done, info = self.env.step(action)
        self._last_obs = observation
        return observation, reward, done, info

    def render(self, *, update_display: bool = False) -> "pygame.surface.Surface":
        if pygame is None:  # pragma: no cover - guarded in tests
            msg = "Pygame is required for rendering."
            raise RuntimeError(msg)
        if self._last_obs is None:
            msg = "Call reset() before render()."
            raise RuntimeError(msg)

        surface = self._surface
        cfg = self.env.config
        surface.fill(self.visual.background_color)

        # Draw pipes.
        for pipe in self.env._pipes:  # type: ignore[attr-defined]
            x = int(float(pipe["x"]) * self.scale)
            gap_center = float(pipe["gap_center"])
            half_gap = cfg.pipe_gap / 2.0
            gap_top = int((gap_center - half_gap) * self.scale)
            gap_bottom = int((gap_center + half_gap) * self.scale)
            width = int(cfg.pipe_width * self.scale)
            pygame.draw.rect(
                surface,
                self.visual.pipe_color,
                pygame.Rect(x, 0, width, gap_top),
            )
            pygame.draw.rect(
                surface,
                self.visual.pipe_color,
                pygame.Rect(x, gap_bottom, width, int(cfg.screen_height * self.scale) - gap_bottom),
            )

        # Draw bird.
        bird_x = int(cfg.bird_x * self.scale)
        bird_y = int(self._last_obs["bird_y"] * self.scale)
        pygame.draw.circle(surface, self.visual.bird_color, (bird_x, bird_y), max(4, int(6 * self.scale)))

        if update_display and self._screen is not None:
            self._screen.blit(surface, (0, 0))
            pygame.display.flip()

        return surface

    def close(self) -> None:
        if pygame is None:  # pragma: no cover - guard
            return
        pygame.display.quit()


__all__ = ["FlappyVisualEnv", "VisualConfig"]
