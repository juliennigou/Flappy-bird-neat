from __future__ import annotations

import math

from games.flappy.env.adapter import action_transform, obs_transform
from games.flappy.env.env_core import FlappyEnv, FlappyEnvConfig


def test_reset_is_deterministic_with_seed() -> None:
    env = FlappyEnv()
    obs_a = env.reset(seed=42)
    env.step(0)
    obs_b = env.reset(seed=42)
    assert obs_a == obs_b


def test_step_updates_position_and_velocity() -> None:
    env = FlappyEnv()
    env.reset(seed=1)
    obs, reward, done, _ = env.step(0)
    assert not done
    assert reward < 0
    assert obs["bird_velocity"] > 0
    env.reset(seed=1)
    obs_flap, _, _, _ = env.step(1)
    assert obs_flap["bird_velocity"] < 0


def test_pipe_pass_reward_accumulates() -> None:
    config = FlappyEnvConfig(pipe_spacing_px=10.0)
    env = FlappyEnv(config)
    env.reset(seed=2)
    env._pipes[0]["x"] = env.config.bird_x - env.config.pipe_width  # type: ignore[index]
    total_reward = 0.0
    for _ in range(200):
        obs, reward, done, info = env.step(0)
        total_reward += reward
        if done:
            break
    assert info["pipes_passed"] > 0
    assert total_reward > config.death_reward


def test_collision_triggers_death_reward() -> None:
    env = FlappyEnv()
    env.reset(seed=0)
    env._bird_y = -10.0  # type: ignore[attr-defined]
    obs, reward, done, _ = env.step(0)
    assert done
    assert math.isclose(reward, env.config.death_reward)
    assert obs is not None


def test_obs_and_action_transforms() -> None:
    env = FlappyEnv()
    obs = env.reset(seed=3)
    vector = obs_transform(obs)
    assert len(vector) == 4
    assert vector[0] == obs["bird_y"]
    assert action_transform([0.4]) == 0
    assert action_transform([0.6]) == 1
