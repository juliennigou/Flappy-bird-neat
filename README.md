# Flappy-bird-neat

Reusable NEAT (NeuroEvolution of Augmenting Topologies) library with a reference Flappy Bird environment. The goal is to provide a generic toolkit that can drive any Pygame project while keeping the NEAT core decoupled from game-specific concerns.

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
```

The `setup` target installs the project in editable mode along with the development toolchain (`ruff`, `black`, `mypy`, `pytest`, `pytest-cov`, `hypothesis`).

## Development workflow

Run the local CI loop before sending changes:

```bash
make lint   # ruff + black --check
make type   # mypy --strict on neatlab/
make test   # pytest with coverage (--maxfail=1, -q)
```

Additional helpers:

- `make format` – apply `black` formatting.
- `make test-all` – verbose pytest run (integration/e2e hooks later).
- `make bench` – placeholder for forthcoming performance benchmarks.

Refer to `agents.md` for the full set of engineering rules and quality gates, and to `plan.md` for the delivery roadmap.
