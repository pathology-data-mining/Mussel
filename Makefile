.PHONY: help install install-dev test test-fast test-slow test-all test-parallel test-watch clean format lint type-check coverage docs build

# Default target - show help
help:
	@echo "Mussel Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make install-gpu      Install with GPU support (CUDA)"
	@echo "  make install-cpu      Install with CPU-only support"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run fast tests only (default, ~6s)"
	@echo "  make test-fast        Run fast tests only (explicit)"
	@echo "  make test-slow        Run slow integration tests (~10min)"
	@echo "  make test-all         Run ALL tests (fast + slow)"
	@echo "  make test-parallel    Run fast tests in parallel (<1s)"
	@echo "  make test-watch       Run tests in watch mode (on file change)"
	@echo "  make test-failed      Re-run only failed tests"
	@echo "  make test-verbose     Run tests with verbose output"
	@echo "  make coverage         Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format           Format code with black and isort"
	@echo "  make format-check     Check formatting without changes"
	@echo "  make lint             Run all linters (black, isort, mypy)"
	@echo "  make type-check       Run type checking with mypy"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean            Remove build artifacts and caches"
	@echo "  make clean-test       Remove test artifacts"
	@echo "  make clean-build      Remove build artifacts"
	@echo "  make clean-pyc        Remove Python cache files"
	@echo ""
	@echo "Build & Distribution:"
	@echo "  make build            Build distribution packages"
	@echo "  make docs             Generate documentation"
	@echo ""
	@echo "Git Helpers:"
	@echo "  make git-status       Show concise git status"
	@echo "  make git-clean-branches  Remove merged branches"
	@echo ""
	@echo "Quick Commands:"
	@echo "  make dev              Install dev dependencies and run fast tests"
	@echo "  make check            Run format check, lint, and tests"
	@echo "  make pre-commit       Run all checks before committing"

# Installation targets
install:
	@echo "Installing Mussel..."
	uv sync

install-dev:
	@echo "Installing Mussel with development dependencies..."
	uv sync --group dev

install-gpu:
	@echo "Installing Mussel with GPU support..."
	uv sync --extra torch-gpu

install-cpu:
	@echo "Installing Mussel with CPU-only support..."
	uv sync --extra torch-cpu

# Testing targets
test:
	@echo "Running fast tests (default)..."
	@uv run pytest tests/

test-fast:
	@echo "Running fast tests only..."
	@uv run pytest tests/ -m "not slow"

test-slow:
	@echo "Running slow integration tests..."
	@uv run pytest tests/ -m slow

test-all:
	@echo "Running ALL tests (fast + slow)..."
	@uv run pytest tests/ -m ""

test-parallel:
	@echo "Running fast tests in parallel..."
	@uv run pytest tests/ -n auto

test-watch:
	@echo "Running tests in watch mode..."
	@uv run pytest tests/ -f

test-failed:
	@echo "Re-running only failed tests..."
	@uv run pytest tests/ --lf

test-verbose:
	@echo "Running tests with verbose output..."
	@uv run pytest tests/ -vv

test-specific:
ifndef TEST
	@echo "Error: Specify test with TEST=path/to/test.py"
	@exit 1
endif
	@echo "Running specific test: $(TEST)"
	@uv run pytest $(TEST) -v

# Coverage
coverage:
	@echo "Running tests with coverage..."
	@uv run pytest tests/ --cov=mussel --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

coverage-report:
	@echo "Opening coverage report..."
	@uv run python -m webbrowser htmlcov/index.html

# Code quality
format:
	@echo "Formatting code..."
	@uv run black mussel/ tests/
	@uv run isort mussel/ tests/
	@echo "Code formatted successfully!"

format-check:
	@echo "Checking code formatting..."
	@uv run black --check mussel/ tests/
	@uv run isort --check mussel/ tests/

lint:
	@echo "Running linters..."
	@uv run black --check mussel/ tests/ || (echo "❌ black failed" && exit 1)
	@uv run isort --check mussel/ tests/ || (echo "❌ isort failed" && exit 1)
	@echo "✅ All linters passed!"

type-check:
	@echo "Running type checker..."
	@uv run mypy mussel/

# Cleaning
clean: clean-build clean-pyc clean-test
	@echo "All clean!"

clean-build:
	@echo "Removing build artifacts..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info
	@rm -rf mussel.egg-info/

clean-pyc:
	@echo "Removing Python cache files..."
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name '__pycache__' -delete
	@find . -type d -name '*.egg-info' -exec rm -rf {} +

clean-test:
	@echo "Removing test artifacts..."
	@rm -rf .pytest_cache/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@rm -rf .mypy_cache/
	@find . -type d -name '.pytest_cache' -exec rm -rf {} +

# Build
build:
	@echo "Building distribution packages..."
	@uv run python -m build

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "Documentation generation not yet configured"

# Git helpers
git-status:
	@echo "Git Status Summary:"
	@echo "==================="
	@git status --short
	@echo ""
	@git branch --show-current | xargs -I {} echo "Current branch: {}"
	@echo ""
	@git log -1 --oneline | xargs -I {} echo "Last commit: {}"

git-clean-branches:
	@echo "Cleaning up merged branches..."
	@git branch --merged | grep -v "\*\|main\|master" | xargs -n 1 git branch -d

# Composite targets
dev: install-dev test-fast
	@echo "✅ Development environment ready!"

check: format-check lint test-fast
	@echo "✅ All checks passed!"

pre-commit: format lint test-fast
	@echo "✅ Ready to commit!"

# Performance profiling
profile-tests:
	@echo "Profiling test performance..."
	@uv run pytest tests/ --durations=20

# Model management
download-models:
	@echo "Pre-downloading models..."
	@uv run python -c "from mussel.models import ModelType; from mussel.models.model_factory import get_model_factory; \
		print('Downloading CLIP...'); \
		get_model_factory(ModelType.CLIP).get_model(None, False, None); \
		print('Downloading ResNet50...'); \
		get_model_factory(ModelType.RESNET50).get_model(None, False, None); \
		print('✅ Models downloaded!')"

# Quick test commands by module
test-models:
	@uv run pytest tests/mussel/models/ -v

test-datasets:
	@uv run pytest tests/mussel/datasets/ -v

test-utils:
	@uv run pytest tests/mussel/utils/ -v

test-cli:
	@uv run pytest tests/mussel/cli/ -v -m "not slow"

test-cli-all:
	@uv run pytest tests/mussel/cli/ -v

# Debug helpers
debug-test:
ifndef TEST
	@echo "Error: Specify test with TEST=path/to/test.py::test_name"
	@exit 1
endif
	@echo "Running test with debugger: $(TEST)"
	@uv run pytest $(TEST) -vv -s --pdb

# Show test markers
test-markers:
	@echo "Available test markers:"
	@uv run pytest --markers | grep "^@pytest.mark" | grep -v "pytest-"

# Show test collection without running
test-collect:
	@echo "Collecting tests..."
	@uv run pytest tests/ --collect-only -q

# Environment info
info:
	@echo "Environment Information:"
	@echo "======================="
	@uv run python --version
	@echo ""
	@uv run pip list | grep -E "(pytest|torch|tensorflow|numpy)" || echo "No ML packages found"
	@echo ""
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "No GPU detected"

# Benchmark tests
benchmark:
	@echo "Running benchmark tests..."
	@uv run pytest tests/ -m "slow" --durations=0 | grep -E "PASSED|FAILED|seconds"
