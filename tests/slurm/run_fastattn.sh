#!/bin/bash
# SLURM job: integration tests for fastattn-based models (Prov-GigaPath).
#
# Submit from the repo root:
#   sbatch tests/slurm/run_fastattn.sh
#
# The fastattn extra uses torch==2.11.0+cu126 and flash-attn 2.6.3 built with
# manylinux_2_28, which is compatible with RHEL 8 (GLIBC 2.28).
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

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"

mkdir -p "$HOME/logs/slurm"
exec > >(tee "$HOME/logs/slurm/test_fastattn_${SLURM_JOB_ID:-local}.log") 2>&1

echo "=== mussel fastattn model tests ==="
echo "Repo:   $REPO_DIR"
echo "Node:   $(hostname)"
echo "GPUs:   ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Python: $(python3 --version 2>&1)"
echo "Date:   $(date)"
echo ""

if [[ -f ~/.hf_cred.env ]]; then
    source ~/.hf_cred.env
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set — prov-gigapath/prov-gigapath is gated and will fail."
fi

export UV_PROJECT_ENVIRONMENT="$HOME/venvs/mussel-fastattn"

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
