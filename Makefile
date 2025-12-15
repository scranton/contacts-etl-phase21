.PHONY: help venv install lint typecheck test qa clean consolidate validate confidence tag pipeline

PYTHON ?= python3.12
VENV_DIR ?= .venv
VENV_BIN := $(VENV_DIR)/bin
PYTHON_BIN := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip
CLI_SCRIPTS := contacts-consolidate contacts-validate contacts-confidence contacts-tag

help:
	@grep -E '^[a-zA-Z_-]+:.*##' Makefile | sort | awk 'BEGIN {FS = ":.*## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

$(PYTHON_BIN):
	$(PYTHON) -m venv $(VENV_DIR)
	$(PYTHON_BIN) -m ensurepip --upgrade

venv: $(PYTHON_BIN) ## Create the virtualenv (if missing) and upgrade pip

install: venv ## Install project with development extras
	@for script in $(CLI_SCRIPTS); do \
		if [ -d "$(VENV_BIN)" ] && [ ! -f "$(VENV_BIN)/$$script" ]; then \
			touch "$(VENV_BIN)/$$script"; \
		fi; \
	done
	$(PYTHON_BIN) -m ensurepip --upgrade
	@$(PYTHON_BIN) -c "import setuptools" >/dev/null 2>&1 || $(PIP) install setuptools wheel
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
