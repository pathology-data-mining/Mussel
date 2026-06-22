#!/bin/bash
# SLURM job: 10-slide neural segmentation correctness validation panel.
#
# Submit from the repo root:
#   sbatch --qos=premium tests/slurm/run_neural_segmentation_correctness_panel.sh

#SBATCH --job-name=mussel-neural-panel
#SBATCH --partition=hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/gpfs/cdsi_ess/home/limr/logs/slurm/neural_seg_panel_%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"

if [[ -f ~/.hf_cred.env ]]; then
    source ~/.hf_cred.env
fi

uv sync --extra torch-gpu

OUT_DIR="/gpfs/cdsi_ess/home/limr/logs/slurm/neural_seg_panel_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

uv run python tests/validation/neural_segmentation_correctness_panel.py \
    --output-dir "$OUT_DIR" \
    --device cuda \
    --batch-size 8
