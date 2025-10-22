"""Command-line interface for NEAT workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    print("Play mode is not implemented yet.", file=sys.stderr)
    return 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    print("Benchmark mode is not implemented yet.", file=sys.stderr)
    return 1


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
        help="Play with a trained checkpoint (not yet implemented)",
    )
    play.set_defaults(func=_cmd_play)

    bench = subparsers.add_parser(
        "benchmark",
        help="Benchmark evaluation performance (not yet implemented)",
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
