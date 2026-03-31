#!/bin/bash
# SLURM job: integration tests for TensorFlow-based models (GooglePath).
#
# Submit from the repo root:
#   sbatch tests/slurm/run_tensorflow.sh
#
# Logs go to ~/logs/slurm/ (writable from all compute nodes).
# A dedicated venv is created at ~/venvs/mussel-tensorflow to avoid conflicts
# with the main torch-gpu environment in the repo.
#
#SBATCH --job-name=mussel-test-tensorflow
#SBATCH --partition=hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

set -euo pipefail

# $SLURM_SUBMIT_DIR is the directory from which sbatch was called (repo root).
# Fallback for running the script interactively.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"

# Redirect output to $HOME which is writable from all compute nodes.
mkdir -p "$HOME/logs/slurm"
exec > >(tee "$HOME/logs/slurm/test_tensorflow_${SLURM_JOB_ID:-local}.log") 2>&1

echo "=== mussel TensorFlow model tests ==="
echo "Repo:   $REPO_DIR"
echo "Node:   $(hostname)"
echo "GPUs:   ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Python: $(python3 --version 2>&1)"
echo "Date:   $(date)"
echo ""

# Load HuggingFace credentials if available
if [[ -f ~/.hf_cred.env ]]; then
    source ~/.hf_cred.env
fi

# Use an isolated venv in $HOME to avoid conflicts with the main torch-gpu
# environment and to ensure the venv is writable from compute nodes.
export UV_PROJECT_ENVIRONMENT="$HOME/venvs/mussel-tensorflow"

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
