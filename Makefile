.PHONY: help install install-dev test test-fast test-slow test-all test-gpu test-fast-gpu test-slow-gpu test-parallel test-watch clean format lint type-check coverage docs build slurm-setup slurm-test-integration slurm-test-fastattn slurm-test-tensorflow slurm-generate-snapshots slurm-test-all slurm-status slurm-logs regression-patch regression-pipeline regression-all docker-build sif

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
	@echo "  make test-gpu         Run ALL tests with GPU acceleration"
	@echo "  make test-fast-gpu    Run fast tests with GPU acceleration"
	@echo "  make test-slow-gpu    Run slow integration tests with GPU acceleration"
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
	@echo "  make docker-build     Build Docker image (mussel:VERSION)"
	@echo "  make sif              Build Apptainer SIF from Docker image (mussel.sif)"
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
	@echo ""
	@echo "SLURM Tests (requires sbatch on an HPC cluster):"
	@echo "  make slurm-setup              Create ~/logs/slurm/ output directory (run once)"
	@echo "  make slurm-test-integration   Submit full integration test array (39 tasks, torch-gpu + fastattn + tensorflow)"
	@echo "  make slurm-test-fastattn      Submit fastattn model tests (GigaPath with flash-attn)"
	@echo "  make slurm-test-tensorflow    Submit TensorFlow model tests (GooglePath)"
	@echo "  make slurm-generate-snapshots Submit golden snapshot generation job"
	@echo "  make slurm-test-all           Submit all SLURM test jobs"
	@echo "  make slurm-status             Show running/pending SLURM jobs for this user"
	@echo "  make slurm-logs               Tail the most recent SLURM log in ~/logs/slurm/"
	@echo ""
	@echo "Regression Tests (requires GPU + reference data on gpfs):"
	@echo "  make regression-patch         Patch-level feature regression vs REEF (OPTIMUS + CTransPath)"
	@echo "  make regression-pipeline      Full pipeline regression (tessellate→extract→filter vs REEF)"
	@echo "  make regression-all           Run both regression tests"

# Installation targets
install:
	@echo "Installing Mussel (CPU torch)..."
	uv sync --extra torch-cpu

install-dev:
	@echo "Installing Mussel with development dependencies..."
	uv sync --extra torch-cpu --group dev

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

test-gpu:
	@echo "Running ALL tests with GPU..."
	@uv run pytest tests/ -m "" --use-gpu --num-workers=8

test-fast-gpu:
	@echo "Running fast tests with GPU..."
	@uv run pytest tests/ -m "not slow" --use-gpu --num-workers=8

test-slow-gpu:
	@echo "Running slow integration tests with GPU..."
	@uv run pytest tests/ -m slow --use-gpu --num-workers=8

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
VERSION := $(shell grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
DOCKER_IMAGE := mussel
SIF_FILE := mussel.sif

build:
	@echo "Building distribution packages..."
	@uv run python -m build

docker-build:
	@echo "Building Docker image $(DOCKER_IMAGE):$(VERSION)..."
	docker build --build-arg BACKEND=torch-gpu -t $(DOCKER_IMAGE):$(VERSION) -t $(DOCKER_IMAGE):latest .
	@echo "✅ Docker image built: $(DOCKER_IMAGE):$(VERSION)"

sif: docker-build
	@echo "Building Apptainer SIF from $(DOCKER_IMAGE):$(VERSION)..."
	apptainer build --force $(SIF_FILE) docker-daemon://$(DOCKER_IMAGE):$(VERSION)
	@echo "✅ SIF built: $(SIF_FILE)"

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

# SLURM test targets
# Requires an HPC cluster with sbatch available.
# Run 'make slurm-setup' once to create the log directory before submitting jobs.

SLURM_LOG_DIR := $(HOME)/logs/slurm
SLURM_SCRIPTS := tests/slurm

slurm-setup:
	@echo "Creating SLURM log directory: $(SLURM_LOG_DIR)"
	@mkdir -p $(SLURM_LOG_DIR)
	@echo "Done — submit jobs with: make slurm-test-integration"

slurm-test-integration: slurm-setup
	@echo "Submitting integration test array (39 tasks)..."
	@sbatch $(SLURM_SCRIPTS)/run_integration.sh
	@echo "Logs: $(SLURM_LOG_DIR)/test_integration_<jobid>_<taskid>.out"

slurm-test-fastattn: slurm-setup
	@echo "Submitting fastattn model tests (GigaPath with flash-attn)..."
	@sbatch $(SLURM_SCRIPTS)/run_fastattn.sh
	@echo "Logs: $(SLURM_LOG_DIR)/test_fastattn_<jobid>.log"

slurm-test-tensorflow: slurm-setup
	@echo "Submitting TensorFlow model tests (GooglePath)..."
	@sbatch $(SLURM_SCRIPTS)/run_tensorflow.sh
	@echo "Logs: $(SLURM_LOG_DIR)/test_tensorflow_<jobid>.log"

slurm-generate-snapshots: slurm-setup
	@echo "Submitting golden snapshot generation job..."
	@sbatch $(SLURM_SCRIPTS)/run_generate_snapshots.sh
	@echo "Logs: $(SLURM_LOG_DIR)/generate_snapshots_<jobid>.out"

slurm-test-all: slurm-test-integration slurm-test-fastattn slurm-test-tensorflow
	@echo "All SLURM test jobs submitted"

slurm-status:
	@squeue --me --format="%.10i %.20j %.8T %.10M %.5D %R" 2>/dev/null || \
		echo "squeue not available — are you on an HPC login node?"

slurm-logs:
	@LATEST=$$(ls -t $(SLURM_LOG_DIR)/*.out $(SLURM_LOG_DIR)/*.log 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "No SLURM logs found in $(SLURM_LOG_DIR)"; \
	else \
		echo "=== $$LATEST ==="; \
		tail -50 "$$LATEST"; \
	fi


# Regression Tests
REGRESSION_SCRIPTS := tests/regression

regression-patch:
	@echo "Running patch-level feature regression (OPTIMUS + CTransPath vs REEF)..."
	@uv run python $(REGRESSION_SCRIPTS)/regression_vs_reference.py

regression-pipeline:
	@echo "Running full pipeline regression (tessellate->extract->filter vs REEF)..."
	@uv run python $(REGRESSION_SCRIPTS)/regression_full_pipeline.py

regression-all: regression-patch regression-pipeline
	@echo "All regression tests complete"
