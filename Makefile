PYTHON ?= python3

.PHONY: setup lint type test test-all bench format

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check neatlab tests
	$(PYTHON) -m black --check neatlab tests

type:
	$(PYTHON) -m mypy neatlab

test:
	$(PYTHON) -m pytest

test-all:
	$(PYTHON) -m pytest -vv

format:
	$(PYTHON) -m black neatlab tests

bench:
	@echo "Benchmarks are not implemented yet."
