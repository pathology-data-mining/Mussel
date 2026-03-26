#!/bin/bash
# SLURM job array: one task per integration test, each gets its own GPU.
#
# Submit from the repo root:
#   sbatch tests/slurm/run_integration.sh
#
# Each task logs to ~/logs/slurm/test_integration_<jobid>_<taskid>_<model>.log
#
#SBATCH --job-name=mussel-test-integration
#SBATCH --partition=hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --array=0-29

set -euo pipefail

# One entry per test — keep in sync with test_encoder_integration.py parametrization.
TESTS=(
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[RESNET50]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[CTRANSPATH]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[GIGAPATH]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[VIRCHOW]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[OPTIMUS]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[CLIP]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[GOOGLEPATH]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[CONCH1_5]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[VIRCHOW2]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[UNI2]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[UNI]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[PHIKON]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[PHIKON_V2]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[H_OPTIMUS_1]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[H0_MINI]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[MIDNIGHT12K]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[GPFM]"
    "tests/mussel/models/test_encoder_integration.py::test_patch_encoder_extracts_features[HIBOU_L]"
    "tests/mussel/models/test_encoder_integration.py::test_slide_encoder_aggregates_features[TITAN_SLIDE]"
    "tests/mussel/models/test_encoder_integration.py::test_slide_encoder_aggregates_features[GIGAPATH_SLIDE]"
    "tests/mussel/models/test_encoder_integration.py::test_slide_encoder_aggregates_features[PRISM_SLIDE]"
    "tests/mussel/models/test_encoder_integration.py::test_slide_encoder_aggregates_features[CHIEF_SLIDE]"
    "tests/mussel/models/test_encoder_integration.py::test_slide_encoder_aggregates_features[FEATHER_SLIDE]"
    "tests/mussel/models/test_encoder_integration.py::test_slide_encoder_aggregates_features[MADELEINE_SLIDE]"
    "tests/mussel/models/test_encoder_integration.py::test_end_to_end_patch_then_slide_encode[CONCH1_5-TITAN_SLIDE]"
    "tests/mussel/models/test_encoder_integration.py::test_end_to_end_patch_then_slide_encode[GIGAPATH-GIGAPATH_SLIDE]"
    "tests/mussel/models/test_encoder_integration.py::test_end_to_end_patch_then_slide_encode[VIRCHOW-PRISM_SLIDE]"
    "tests/mussel/models/test_encoder_integration.py::test_end_to_end_patch_then_slide_encode[CTRANSPATH-CHIEF_SLIDE]"
    "tests/mussel/models/test_encoder_integration.py::test_end_to_end_patch_then_slide_encode[CONCH1_5-FEATHER_SLIDE]"
    "tests/mussel/models/test_encoder_integration.py::test_end_to_end_patch_then_slide_encode[CLIP-MADELEINE_SLIDE]"
)

TEST_NODE="${TESTS[$SLURM_ARRAY_TASK_ID]}"
LABEL=$(echo "$TEST_NODE" | sed 's/.*\[//;s/\]//')

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"

mkdir -p "$HOME/logs/slurm"
exec > >(tee "$HOME/logs/slurm/test_integration_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${LABEL}.log") 2>&1

echo "=== mussel integration test: ${LABEL} ==="
echo "Repo:   $REPO_DIR"
echo "Node:   $(hostname)"
echo "GPUs:   ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Python: $(python3 --version 2>&1)"
echo "Date:   $(date)"
echo ""

if [[ -f ~/.hf_cred.env ]]; then
    source ~/.hf_cred.env
fi

uv sync --extra torch-gpu

echo ""
echo "--- Running: ${TEST_NODE} ---"
uv run pytest "$TEST_NODE" \
    --use-gpu \
    -v \
    --tb=short \
    --timeout=300

EXIT_CODE=$?
echo ""
echo "=== Test finished with exit code ${EXIT_CODE} ==="
exit $EXIT_CODE
