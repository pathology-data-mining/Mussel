#!/bin/bash
# SLURM job: integration tests for fastattn-based models (Prov-GigaPath slide encoder).
#
# Submit from the repo root:
#   sbatch tests/slurm/run_fastattn.sh
#
# On first run the job will compile flash-attn from source (~15 min) if the
# pre-built wheel is incompatible with the cluster's GLIBC.  Subsequent runs
# reuse the compiled .so and skip compilation entirely.
#
# Logs go to ~/logs/slurm/ (writable from all compute nodes).
#
#SBATCH --job-name=mussel-test-fastattn
#SBATCH --partition=hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00

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
# Allow flash-attn source compilation in parallel
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-4}"

echo "--- Installing fastattn extra into ${UV_PROJECT_ENVIRONMENT} ---"
uv sync --extra fastattn

# ---------------------------------------------------------------------------
# flash-attn source compilation fallback
#
# Pre-built wheels from the lock file target glibc >= 2.32 (Ubuntu 20.04+).
# RHEL 8 nodes have glibc 2.28.  If the binary can't load, recompile it here
# using the system CUDA toolkit and GCC; the resulting .so will be linked
# against glibc 2.28 and will persist in the venv for future runs.
# ---------------------------------------------------------------------------
VENV_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
VENV_PIP="${UV_PROJECT_ENVIRONMENT}/bin/pip"

if ! "$VENV_PYTHON" -c "import flash_attn" 2>/dev/null; then
    echo ""
    echo "--- Pre-built flash-attn incompatible (likely GLIBC mismatch); compiling from source ---"
    echo "    MAX_JOBS=${MAX_JOBS}  (parallel ninja workers)"

    # Prefer the highest available CUDA version so the compiled extension
    # is compatible with the runtime driver.
    for CUDA_VER in 12.3 12 11.8; do
        if [[ -x "/usr/local/cuda-${CUDA_VER}/bin/nvcc" ]]; then
            export CUDA_HOME="/usr/local/cuda-${CUDA_VER}"
            break
        fi
    done
    export PATH="${CUDA_HOME}/bin:${PATH}"
    echo "    CUDA_HOME=${CUDA_HOME}"
    nvcc --version | head -1

    # Install flash-attn from source.  --no-build-isolation uses the venv's
    # torch so setup.py can probe the CUDA arch list automatically.
    "$VENV_PIP" install "flash-attn==2.5.9" \
        --no-build-isolation \
        --force-reinstall \
        --no-binary :all: \
        --verbose
    echo "--- flash-attn compiled from source ---"
fi

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
