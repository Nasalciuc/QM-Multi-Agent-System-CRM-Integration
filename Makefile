# =============================================================================
# Makefile — QM Multi-Agent System
# =============================================================================
# Cross-platform targets for common development tasks.
# Usage: make <target>
#
# Requires: Python 3.11+, pip, Docker (for container targets)
# =============================================================================

.PHONY: help install install-dev test lint format docker-build docker-test docker-up docker-down clean lock

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Python ──────────────────────────────────────────────────────

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: install ## Install production + dev dependencies
	pip install -r requirements-dev.txt

test: ## Run tests with coverage
	python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint: ## Run ruff linter
	ruff check src/ tests/

format: ## Auto-format with ruff
	ruff format src/ tests/

lock: ## Generate requirements lockfile (#19)
	pip freeze --exclude-editable > requirements.lock

# ── Docker ──────────────────────────────────────────────────────

docker-build: ## Build production Docker image
	docker build --target production -t qm-system .

docker-test: ## Build and run tests in Docker
	docker build --target test -t qm-system-test .
	docker run --rm qm-system-test

docker-up: ## Start services with docker-compose
	docker compose up -d

docker-down: ## Stop services
	docker compose down

# ── Cleanup ─────────────────────────────────────────────────────

clean: ## Remove caches, __pycache__, .pyc files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ dist/ build/ *.egg-info/
