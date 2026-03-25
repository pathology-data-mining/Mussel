#!/bin/bash
# SLURM job: integration tests for TensorFlow-based models (GooglePath).
#
# Submit from the repo root:
#   sbatch tests/slurm/test_tensorflow.sh
#
#SBATCH --job-name=mussel-test-tensorflow
#SBATCH --partition=hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm/test_tensorflow_%j.out
#SBATCH --error=logs/slurm/test_tensorflow_%j.err

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs/slurm
echo "Node:   $(hostname)"
echo "GPUs:   ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Python: $(python3 --version 2>&1)"
echo "Date:   $(date)"
echo ""

# Load HuggingFace credentials if available
if [[ -f ~/.hf_cred.env ]]; then
    source ~/.hf_cred.env
fi

# Use an isolated venv to avoid conflicts with the main torch-gpu environment
export UV_PROJECT_ENVIRONMENT=".venv-tensorflow"

echo "--- Installing tensorflow-gpu extra into ${UV_PROJECT_ENVIRONMENT} ---"
uv sync --extra tensorflow-gpu

echo ""
echo "--- Running TensorFlow model tests ---"
uv run pytest tests/mussel/models/test_tensorflow_models.py \
    --use-gpu \
    -v \
    --tb=short \
    -m requires_tensorflow

EXIT_CODE=$?
echo ""
echo "=== Tests finished with exit code ${EXIT_CODE} ==="
exit $EXIT_CODE
