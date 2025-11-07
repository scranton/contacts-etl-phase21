.PHONY: help venv install lint typecheck test qa clean consolidate validate confidence tag pipeline

PYTHON ?= python3.12
VENV_DIR ?= .venv
VENV_BIN := $(VENV_DIR)/bin
PYTHON_BIN := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip

help:
	@grep -E '^[a-zA-Z_-]+:.*##' Makefile | sort | awk 'BEGIN {FS = ":.*## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

$(PYTHON_BIN):
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip

venv: $(PYTHON_BIN) ## Create the virtualenv (if missing) and upgrade pip

install: venv ## Install project with development extras
	$(PIP) install --no-build-isolation -e ".[dev]"

lint: install ## Run static lint checks
	$(VENV_BIN)/flake8 src tests

typecheck: install ## Run mypy type checks
	$(VENV_BIN)/mypy src

test: install ## Execute the test suite
	$(VENV_BIN)/pytest

qa: lint typecheck test ## Run the full quality gate (lint, type, tests)

clean: ## Remove virtualenv and caches
	rm -rf $(VENV_DIR) .pytest_cache .mypy_cache .coverage

consolidate: install ## Run contacts-consolidate pipeline stage
	$(VENV_BIN)/contacts-consolidate --config config.yaml

validate: install ## Run contacts-validate pipeline stage
	$(VENV_BIN)/contacts-validate --config config.yaml

confidence: install ## Run contacts-confidence pipeline stage
	$(VENV_BIN)/contacts-confidence --config config.yaml

tag: install ## Run contacts-tag pipeline stage
	$(VENV_BIN)/contacts-tag --config config.yaml

pipeline: consolidate validate confidence tag ## Execute all pipeline stages in order
