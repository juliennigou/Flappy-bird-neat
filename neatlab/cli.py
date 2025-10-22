"""Command-line interface for NEAT workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from games.flappy.config import load_env_config
from games.flappy.env.env_core import FlappyEnv
from games.flappy.env.env_pygame import FlappyVisualEnv

from .benchmark import run_benchmark
from .config import NEATConfig, RunConfig, load_neat_config, load_run_config
from .training import run_training


def _load_bundle(config_path: Path) -> tuple[RunConfig, NEATConfig]:
    run_config = load_run_config(config_path)
    neat_config = load_neat_config(run_config.neat_config)
    return run_config, neat_config


def _cmd_train(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    run_config, neat_config = _load_bundle(config_path)

    if args.dry_run:
        print("[train] configuration validated")
        print(f"  neat_config: {run_config.neat_config}")
        print(f"  env_config: {run_config.env_config}")
        print(f"  population_size: {neat_config.population_size}")
        print(f"  workers: {run_config.workers}")
        return 0

    run_training(run_config, neat_config)
    return 0


def _cmd_play(args: argparse.Namespace) -> int:
    env_path = Path(args.env_config)
    if args.dry_run:
        load_env_config(env_path)
        print("[play] configuration validated")
        return 0

    return _play_interactive(env_path, args.seed, args.fps)


def _play_interactive(
    env_path: Path,
    seed: int | None,
    fps: int,
) -> int:  # pragma: no cover - requires Pygame window
    try:
        import pygame
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        print("Pygame is required for play mode.", file=sys.stderr)
        raise SystemExit(1) from exc

    config = load_env_config(env_path)
    headless = FlappyEnv(config)
    visual = FlappyVisualEnv(env=headless, init_pygame=True)
    visual.reset(seed=seed)

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        pressed = pygame.key.get_pressed()
        action = 1 if pressed[pygame.K_SPACE] else 0
        _observation, reward, done, _ = visual.step(action)
        visual.render(update_display=True)
        clock.tick(max(fps, 1))

        if done:
            print(
                "Episode finished (reward="
                f"{reward}). Press ESC to exit or SPACE to restart."
            )
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            waiting = False
                            break
                        if event.key == pygame.K_SPACE:
                            visual.reset(seed=seed)
                            waiting = False
                            break
                if not waiting:
                    break
            continue

    visual.close()
    pygame.quit()
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    env_path = Path(args.env_config)
    env_config = load_env_config(env_path)
    raw_workers = tuple(args.workers)
    if any(worker <= 0 for worker in raw_workers):
        print("Worker counts must be positive integers.", file=sys.stderr)
        return 1
    workers = tuple(sorted(set(raw_workers)))

    if args.dry_run:
        print("[benchmark] configuration validated")
        print(f"  env_config: {env_path}")
        print(f"  workers: {workers}")
        print(f"  steps_target: {args.steps}")
        print(f"  population_size: {args.population_size}")
        print(f"  episodes_per_genome: {args.episodes}")
        return 0

    results = run_benchmark(
        env_config,
        steps_target=args.steps,
        population_size=args.population_size,
        episodes_per_genome=args.episodes,
        worker_counts=workers,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(f"[benchmark] env={env_path}")
    for item in results:
        throughput = int(item.steps_per_sec)
        duration = item.duration_s
        print(
            "  workers={workers} steps={steps} duration={duration:.2f}s "
            "throughput={throughput} steps/s "
            "(iterations={iterations})".format(
                workers=item.workers,
                steps=item.steps,
                duration=duration,
                throughput=throughput,
                iterations=item.iterations,
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neatlab",
        description="NEAT command-line interface",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train",
        help="Run training using a YAML configuration bundle",
    )
    train.add_argument(
        "--config",
        required=True,
        help="Path to run configuration YAML",
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training",
    )
    train.set_defaults(func=_cmd_train)

    play = subparsers.add_parser(
        "play",
        help="Play Flappy Bird manually using the Pygame renderer",
    )
    play.add_argument("--env-config", required=True, help="Path to Flappy env YAML")
    play.add_argument("--seed", type=int, default=None, help="Initial environment seed")
    play.add_argument("--fps", type=int, default=60, help="Target frames per second")
    play.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without launching the renderer",
    )
    play.set_defaults(func=_cmd_play)

    bench = subparsers.add_parser(
        "benchmark",
        help="Benchmark evaluation performance.",
    )
    bench.add_argument(
        "--env-config",
        default="games/flappy/configs/env.yml",
        help="Path to Flappy environment configuration YAML.",
    )
    bench.add_argument(
        "--steps",
        type=int,
        default=200_000,
        help="Target number of environment steps per measurement.",
    )
    bench.add_argument(
        "--population-size",
        type=int,
        default=32,
        help="Number of genomes evaluated per batch.",
    )
    bench.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Episodes per genome for benchmark evaluations.",
    )
    bench.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Worker counts to benchmark (space-separated list).",
    )
    bench.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size supplied to the parallel evaluator.",
    )
    bench.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic benchmarks.",
    )
    bench.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate benchmark configuration without executing it.",
    )
    bench.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    return int(result)


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
