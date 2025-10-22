# Contributing

Thanks for your interest in improving **Flappy-bird-neat**! The project follows an incremental process focused on safety, reproducibility, and maintainability. Before opening a pull request, please read the key guidelines below.

## Development workflow

1. Read the relevant specs (`agents.md`, `plan.md`, and the technical specification).
2. Capture the current objective in a short checklist before coding.
3. Start with tests (unit, integration, or property-based) that describe the behaviour you want.
4. Implement the minimal code necessary to satisfy the tests.
5. Run the full local CI suite before committing:
   - `make lint`
   - `make type`
   - `make test`
6. Keep changes small and focused. One objective per pull request.

## Coding standards

- Python 3.10+, with strict typing (`mypy --strict` on `neatlab/`).
- Formatting via `black`; linting via `ruff`.
- Google-style docstrings with clear rationale for non-trivial logic.
- Avoid coupling `neatlab/` to Pygame or any environment-specific resources.
- Design for deterministic, seedable runs (see `agents.md` for RNG rules).

## Tests & benchmarks

- Unit tests must accompany every new feature.
- Target ≥85 % coverage on `neatlab/`; integrate with `pytest --cov`.
- Add focused performance benchmarks when touching hot paths (`make bench`).

## Commit conventions

- Use Conventional Commits (e.g. `feat(core): add innovation tracker snapshots`).
- Reference issues or plan steps when relevant.

## Reporting issues

Please include:

- Steps to reproduce.
- Expected versus actual behaviour.
- Environment details (OS, Python version, relevant configs).

We appreciate contributions that follow the roadmap laid out in `plan.md`. Thanks for helping build a reusable NEAT toolkit!
