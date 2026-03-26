#!/bin/bash
# SLURM job: integration tests for all torch-gpu models.
#
# Submit from the repo root:
#   sbatch tests/slurm/run_integration.sh
#
# Logs go to ~/logs/slurm/ (writable from all compute nodes).
#
#SBATCH --job-name=mussel-test-integration
#SBATCH --partition=hpc
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"

mkdir -p "$HOME/logs/slurm"
exec > >(tee "$HOME/logs/slurm/test_integration_${SLURM_JOB_ID:-local}.log") 2>&1

echo "=== mussel integration tests ==="
echo "Repo:   $REPO_DIR"
echo "Node:   $(hostname)"
echo "GPUs:   ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Python: $(python3 --version 2>&1)"
echo "Date:   $(date)"
echo ""

if [[ -f ~/.hf_cred.env ]]; then
    source ~/.hf_cred.env
fi

echo "--- Installing torch-gpu extra ---"
uv sync --extra torch-gpu

echo ""
echo "--- Running integration tests ---"
uv run pytest tests/mussel/models/test_encoder_integration.py \
    --use-gpu \
    -v \
    --tb=short \
    --timeout=600

EXIT_CODE=$?
echo ""
echo "=== Tests finished with exit code ${EXIT_CODE} ==="
exit $EXIT_CODE
