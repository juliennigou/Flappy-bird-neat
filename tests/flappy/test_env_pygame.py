from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

if os.environ.get("PYGAME_TESTS", "0") != "1":
    pytest.skip("Pygame visual tests disabled", allow_module_level=True)

pytest.importorskip("pygame")
import pygame
from games.flappy.env.env_core import FlappyEnv
from games.flappy.env.env_pygame import FlappyVisualEnv


def test_flappy_visual_env_renders_surface() -> None:
    env = FlappyVisualEnv(env=FlappyEnv(), init_pygame=False)
    env.reset(seed=1)
    surface = env.render()
    assert surface.get_width() > 0
    assert surface.get_height() > 0
    env.close()


def test_flappy_visual_env_updates_display() -> None:
    env = FlappyVisualEnv(env=FlappyEnv(), init_pygame=True)
    env.reset(seed=1)
    env.render(update_display=True)
    env.close()
    assert pygame.display.get_surface() is None
