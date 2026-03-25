#!/bin/bash
# SLURM job: integration tests for fastattn-based models (Prov-GigaPath slide encoder).
#
# Submit from the repo root:
#   sbatch tests/slurm/run_fastattn.sh
#
# Note: fastattn pins torch==2.1.2 and requires flash-attn compiled for your
# CUDA version. A dedicated venv is created at ~/venvs/mussel-fastattn to
# avoid conflicts with the main torch-gpu environment.
#
# Logs go to ~/logs/slurm/ (writable from all compute nodes).
#
#SBATCH --job-name=mussel-test-fastattn
#SBATCH --partition=hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

set -euo pipefail

# $SLURM_SUBMIT_DIR is the directory from which sbatch was called (repo root).
# Fallback for running the script interactively.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"

# Redirect output to $HOME which is writable from all compute nodes.
mkdir -p "$HOME/logs/slurm"
exec > >(tee "$HOME/logs/slurm/test_fastattn_${SLURM_JOB_ID:-local}.log") 2>&1

echo "=== mussel fastattn model tests ==="
echo "Repo:   $REPO_DIR"
echo "Node:   $(hostname)"
echo "GPUs:   ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Python: $(python3 --version 2>&1)"
echo "Date:   $(date)"
echo ""

# Load HuggingFace credentials (required: prov-gigapath/prov-gigapath is gated)
if [[ -f ~/.hf_cred.env ]]; then
    source ~/.hf_cred.env
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set — prov-gigapath/prov-gigapath is gated and will fail."
fi

# Use an isolated venv in $HOME to avoid conflicts with the main torch-gpu
# environment and to ensure the venv is writable from compute nodes.
# flash-attn must be compiled for the CUDA version on this node; set
# FLASH_ATTENTION_SKIP_CUDA_BUILD=0 to allow recompilation if needed.
export UV_PROJECT_ENVIRONMENT="$HOME/venvs/mussel-fastattn"
export FLASH_ATTENTION_SKIP_CUDA_BUILD="${FLASH_ATTENTION_SKIP_CUDA_BUILD:-0}"

echo "--- Installing fastattn extra into ${UV_PROJECT_ENVIRONMENT} ---"
uv sync --extra fastattn

echo ""
echo "--- Running fastattn model tests ---"
uv run pytest tests/mussel/models/test_fastattn_models.py \
    --use-gpu \
    -v \
    --tb=short \
    -m requires_fastattn

EXIT_CODE=$?
echo ""
echo "=== Tests finished with exit code ${EXIT_CODE} ==="
exit $EXIT_CODE
