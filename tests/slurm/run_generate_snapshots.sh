#!/bin/bash
# SLURM job: generate golden snapshot .npy files for all patch encoders.
#
# Run this ONCE to populate tests/testdata/snapshots/ before running
# snapshot regression tests. Snapshots are committed to the repo so
# subsequent CI runs can compare against them.
#
# Submit from the repo root:
#   sbatch tests/slurm/run_generate_snapshots.sh
#
#SBATCH --job-name=mussel-generate-snapshots
#SBATCH --partition=hpc
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/cdsi_ess/home/limr/logs/slurm/generate_snapshots_%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"

echo "=== mussel: generate golden snapshots ==="
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
echo "--- Generating snapshots (torch-gpu models) ---"
uv run pytest tests/mussel/models/test_encoder_integration.py \
    -k "test_patch_encoder_matches_snapshot" \
    --use-gpu \
    --update-snapshots \
    -v \
    --tb=short \
    --timeout=600

EXIT_CODE=$?
echo ""
echo "Snapshots written to: tests/testdata/snapshots/"
ls -lh tests/testdata/snapshots/*.npy 2>/dev/null || echo "(none written)"
echo ""
echo "=== Finished with exit code ${EXIT_CODE} ==="
exit $EXIT_CODE
