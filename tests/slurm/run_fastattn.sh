#!/bin/bash
# SLURM job: integration tests for fastattn-based models (Prov-GigaPath slide encoder).
#
# Submit from the repo root:
#   sbatch tests/slurm/test_fastattn.sh
#
# Note: fastattn pins torch==2.1.2 and requires flash-attn compiled for your
# CUDA version. The job installs into an isolated venv (.venv-fastattn) to
# avoid conflicts with the main torch-gpu environment.
#
#SBATCH --job-name=mussel-test-fastattn
#SBATCH --partition=hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/test_fastattn_%j.out
#SBATCH --error=logs/slurm/test_fastattn_%j.err

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs/slurm
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

# Use an isolated venv to avoid conflicts with the main torch-gpu environment.
# flash-attn must be compiled for the CUDA version on this node; set
# FLASH_ATTENTION_SKIP_CUDA_BUILD=0 to allow recompilation if needed.
export UV_PROJECT_ENVIRONMENT=".venv-fastattn"
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
