"""Pygame-based renderer for the Flappy headless environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .env_core import FlappyEnv, FlappyEnvConfig

pygame: Any | None = None

if TYPE_CHECKING:  # pragma: no cover - typing references only
    import pygame.surface


def _load_pygame() -> Any:
    global pygame
    if pygame is None:  # pragma: no cover - optional dependency path
        try:
            import pygame as _pygame
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            msg = "Pygame is required for FlappyVisualEnv."
            raise RuntimeError(msg) from exc
        pygame = _pygame
    return pygame


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
        _pg = _load_pygame()

        self.env = env or FlappyEnv(env_config)
        self.visual = visual_config or VisualConfig()
        self.scale = self.visual.scale
        self._screen: "pygame.surface.Surface" | None = None
        self._last_obs: dict[str, float] | None = None

        width = int(self.env.config.screen_width * self.scale)
        height = int(self.env.config.screen_height * self.scale)
        self._surface = _pg.Surface((width, height))

        if init_pygame:
            _pg.display.init()
            self._screen = _pg.display.set_mode((width, height))
            _pg.display.set_caption("Flappy Bird (visual)")

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
        if self._last_obs is None:
            msg = "Call reset() before render()."
            raise RuntimeError(msg)

        surface = self._surface
        cfg = self.env.config
        surface.fill(self.visual.background_color)

        _pg = _load_pygame()

        # Draw pipes.
        for pipe in self.env._pipes:  # type: ignore[attr-defined]
            x = int(float(pipe["x"]) * self.scale)
            gap_center = float(pipe["gap_center"])
            half_gap = cfg.pipe_gap / 2.0
            gap_top = int((gap_center - half_gap) * self.scale)
            gap_bottom = int((gap_center + half_gap) * self.scale)
            width = int(cfg.pipe_width * self.scale)
            _pg.draw.rect(
                surface,
                self.visual.pipe_color,
                _pg.Rect(x, 0, width, gap_top),
            )
            _pg.draw.rect(
                surface,
                self.visual.pipe_color,
                _pg.Rect(
                    x,
                    gap_bottom,
                    width,
                    int(cfg.screen_height * self.scale) - gap_bottom,
                ),
            )

        # Draw bird.
        bird_x = int(cfg.bird_x * self.scale)
        bird_y = int(self._last_obs["bird_y"] * self.scale)
        _pg.draw.circle(
            surface,
            self.visual.bird_color,
            (bird_x, bird_y),
            max(4, int(6 * self.scale)),
        )

        if update_display and self._screen is not None:
            self._screen.blit(surface, (0, 0))
            _pg.display.flip()

        return surface

    def close(self) -> None:
        try:
            _pg = _load_pygame()
        except RuntimeError:  # pragma: no cover - pygame not installed
            return
        _pg.display.quit()


__all__ = ["FlappyVisualEnv", "VisualConfig"]
