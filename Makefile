# Makefile for Surrogate Gradient Optimization Lab

# =============================================================================
# Configuration
# =============================================================================
.PHONY: help install install-dev test lint format type-check security clean build docker run docs deploy

# Default Python interpreter
PYTHON ?= python3
PIP ?= pip3

# Project configuration
PROJECT_NAME = surrogate-gradient-optim-lab
PACKAGE_NAME = surrogate_optim
VERSION = $(shell $(PYTHON) -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")

# Docker configuration
DOCKER_IMAGE = $(PROJECT_NAME)
DOCKER_TAG = $(VERSION)
DOCKER_REGISTRY = ghcr.io/terragon-labs

# Test configuration
TEST_PATH = tests/
MIN_COVERAGE = 80

# Documentation
DOCS_DIR = docs/
DOCS_BUILD_DIR = $(DOCS_DIR)_build/

# =============================================================================
# Help
# =============================================================================
help: ## Show this help message
	@echo "Surrogate Gradient Optimization Lab - Makefile Commands"
	@echo "======================================================"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make install-dev     # Install development dependencies"
	@echo "  make test           # Run full test suite"
	@echo "  make lint           # Run linting checks"
	@echo "  make docker-build   # Build Docker image"
	@echo ""

# =============================================================================
# Installation
# =============================================================================
install: ## Install package for production
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install .

install-dev: ## Install package with development dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,docs,benchmark,notebook]"
	pre-commit install

install-gpu: ## Install package with GPU support
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[gpu,dev]"

# =============================================================================
# Code Quality
# =============================================================================
lint: ## Run linting checks
	@echo "Running linting checks..."
	ruff check $(PACKAGE_NAME)/ tests/
	flake8 $(PACKAGE_NAME)/ tests/
	mypy $(PACKAGE_NAME)/

format: ## Format code
	@echo "Formatting code..."
	black $(PACKAGE_NAME)/ tests/ examples/
	isort $(PACKAGE_NAME)/ tests/ examples/
	ruff check --fix $(PACKAGE_NAME)/ tests/

format-check: ## Check code formatting without making changes
	@echo "Checking code formatting..."
	black --check $(PACKAGE_NAME)/ tests/ examples/
	isort --check-only $(PACKAGE_NAME)/ tests/ examples/

type-check: ## Run type checking
	@echo "Running type checks..."
	mypy $(PACKAGE_NAME)/

security: ## Run security checks
	@echo "Running security checks..."
	bandit -r $(PACKAGE_NAME)/ -x tests/
	safety check

pre-commit: ## Run all pre-commit hooks
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files

# =============================================================================
# Testing
# =============================================================================
test: ## Run all tests
	@echo "Running test suite..."
	pytest $(TEST_PATH) -v --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term-missing --cov-fail-under=$(MIN_COVERAGE)

test-fast: ## Run fast tests only
	@echo "Running fast tests..."
	pytest $(TEST_PATH) -v -m "not slow" --cov=$(PACKAGE_NAME) --cov-report=term-missing

test-slow: ## Run slow tests only
	@echo "Running slow tests..."
	pytest $(TEST_PATH) -v -m "slow"

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	pytest $(TEST_PATH)integration/ -v

test-benchmark: ## Run benchmark tests
	@echo "Running benchmark tests..."
	pytest $(TEST_PATH)benchmarks/ -v -m "benchmark" --benchmark-only

test-parallel: ## Run tests in parallel
	@echo "Running tests in parallel..."
	pytest $(TEST_PATH) -v -n auto --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term-missing

test-watch: ## Run tests in watch mode
	@echo "Running tests in watch mode..."
	ptw -- $(TEST_PATH) -v

coverage: ## Generate coverage report
	@echo "Generating coverage report..."
	pytest $(TEST_PATH) --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/"

# =============================================================================
# Build and Package
# =============================================================================
build: clean ## Build package
	@echo "Building package..."
	$(PYTHON) -m build

build-wheel: ## Build wheel only
	@echo "Building wheel..."
	$(PYTHON) -m build --wheel

build-sdist: ## Build source distribution only
	@echo "Building source distribution..."
	$(PYTHON) -m build --sdist

publish-test: build ## Publish to test PyPI
	@echo "Publishing to test PyPI..."
	twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	@echo "Publishing to PyPI..."
	twine upload dist/*

# =============================================================================
# Docker
# =============================================================================
docker-build: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_IMAGE):latest

docker-build-dev: ## Build development Docker image
	@echo "Building development Docker image..."
	docker build --target development -t $(DOCKER_IMAGE):dev .

docker-build-gpu: ## Build GPU Docker image
	@echo "Building GPU Docker image..."
	docker build --target gpu -t $(DOCKER_IMAGE):gpu .

docker-run: ## Run Docker container
	@echo "Running Docker container..."
	docker run -it --rm -p 8000:8000 $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run-dev: ## Run development Docker container
	@echo "Running development Docker container..."
	docker run -it --rm -v $(PWD):/workspace -p 8888:8888 -p 8080:8080 $(DOCKER_IMAGE):dev

docker-push: ## Push Docker image to registry
	@echo "Pushing Docker image to registry..."
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker tag $(DOCKER_IMAGE):latest $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):latest
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):latest

# =============================================================================
# Docker Compose
# =============================================================================
up: ## Start all services with docker-compose
	@echo "Starting services..."
	docker-compose up -d

up-dev: ## Start development services
	@echo "Starting development services..."
	docker-compose up -d surrogate-optim-dev

up-gpu: ## Start GPU services
	@echo "Starting GPU services..."
	docker-compose --profile gpu up -d

down: ## Stop all services
	@echo "Stopping services..."
	docker-compose down

logs: ## Show logs from all services
	@echo "Showing logs..."
	docker-compose logs -f

# =============================================================================
# Documentation
# =============================================================================
docs: ## Build documentation
	@echo "Building documentation..."
	cd $(DOCS_DIR) && make html

docs-serve: ## Serve documentation locally
	@echo "Serving documentation..."
	cd $(DOCS_BUILD_DIR)html && $(PYTHON) -m http.server 8080

docs-clean: ## Clean documentation build
	@echo "Cleaning documentation..."
	cd $(DOCS_DIR) && make clean

docs-linkcheck: ## Check documentation links
	@echo "Checking documentation links..."
	cd $(DOCS_DIR) && make linkcheck

# =============================================================================
# Development
# =============================================================================
dev-setup: ## Set up development environment
	@echo "Setting up development environment..."
	$(MAKE) install-dev
	pre-commit install
	@echo "Development environment ready!"

dev-update: ## Update development dependencies
	@echo "Updating development dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,docs,benchmark,notebook]" --upgrade
	pre-commit autoupdate

jupyter: ## Start Jupyter Lab
	@echo "Starting Jupyter Lab..."
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

notebook: ## Start Jupyter Notebook
	@echo "Starting Jupyter Notebook..."
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# =============================================================================
# Data and Models
# =============================================================================
download-data: ## Download sample datasets
	@echo "Downloading sample datasets..."
	mkdir -p data/
	# Add commands to download datasets

generate-data: ## Generate synthetic datasets
	@echo "Generating synthetic datasets..."
	$(PYTHON) -m $(PACKAGE_NAME).data.generate_synthetic

# =============================================================================
# Benchmarks
# =============================================================================
benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	$(PYTHON) -m $(PACKAGE_NAME).benchmarks.run_all

benchmark-report: ## Generate benchmark report
	@echo "Generating benchmark report..."
	$(PYTHON) -m $(PACKAGE_NAME).benchmarks.generate_report

# =============================================================================
# Maintenance
# =============================================================================
clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean ## Clean all artifacts including Docker
	@echo "Cleaning all artifacts..."
	docker system prune -f
	docker volume prune -f

update-deps: ## Update dependencies
	@echo "Updating dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,docs,benchmark,notebook]" --upgrade

check-deps: ## Check for outdated dependencies
	@echo "Checking for outdated dependencies..."
	$(PIP) list --outdated

# =============================================================================
# CI/CD
# =============================================================================
ci-test: ## Run CI test suite
	@echo "Running CI test suite..."
	$(MAKE) lint
	$(MAKE) security
	$(MAKE) test

ci-build: ## Build for CI
	@echo "Building for CI..."
	$(MAKE) build
	$(MAKE) docker-build

# =============================================================================
# Release Management
# =============================================================================
version: ## Show current version
	@echo "Current version: $(VERSION)"

version-bump-patch: ## Bump patch version
	@echo "Bumping patch version..."
	bump2version patch

version-bump-minor: ## Bump minor version
	@echo "Bumping minor version..."
	bump2version minor

version-bump-major: ## Bump major version
	@echo "Bumping major version..."
	bump2version major

release: ## Create a release
	@echo "Creating release..."
	$(MAKE) clean
	$(MAKE) ci-test
	$(MAKE) build
	$(MAKE) docker-build
	@echo "Release ready for deployment!"

# =============================================================================
# Info
# =============================================================================
info: ## Show project information
	@echo "Project Information"
	@echo "==================="
	@echo "Name: $(PROJECT_NAME)"
	@echo "Package: $(PACKAGE_NAME)"
	@echo "Version: $(VERSION)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Docker: $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo ""

# Set default target
.DEFAULT_GOAL := help