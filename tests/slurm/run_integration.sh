#!/bin/bash
# SLURM job array: one task per model integration test.
# Covers torch-gpu, fastattn, and tensorflow extras in a single array.
#
# One-time setup (output dir must exist before submitting):
#   mkdir -p ~/logs/slurm
#
# Submit from the repo root:
#   sbatch tests/slurm/run_integration.sh
#
# Each task logs to ~/logs/slurm/test_integration_<jobid>_<taskid>.out
#
#SBATCH --job-name=mussel-test-integration
#SBATCH --partition=hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --array=0-37
#SBATCH --output=/gpfs/cdsi_ess/home/limr/logs/slurm/test_integration_%A_%a.out

set -euo pipefail

# Parallel arrays: test node ID, uv extra, and optional venv override.
# Venv is empty for torch-gpu (uses repo .venv); set for fastattn/tensorflow
# to avoid conflicts with the torch-gpu environment.
TESTS=(
    # --- torch-gpu (tasks 0-29) ---
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
    # --- fastattn (tasks 30-32) ---
    "tests/mussel/models/test_fastattn_models.py::test_gigapath_patch_encoder_extracts_features"
    "tests/mussel/models/test_fastattn_models.py::test_gigapath_slide_encoder_aggregates_features"
    "tests/mussel/models/test_fastattn_models.py::test_gigapath_end_to_end"
    # --- tensorflow (task 33) ---
    "tests/mussel/models/test_tensorflow_models.py::test_tensorflow_patch_encoder_extracts_features[GOOGLEPATH]"
    # --- neural segmentation + artifact removal (tasks 34-37) ---
    "tests/mussel/utils/test_segmentation_integration.py::test_neural_segmentation_produces_valid_patches"
    "tests/mussel/utils/test_segmentation_integration.py::test_neural_segmentation_patch_count_close_to_hsv"
    "tests/mussel/utils/test_segmentation_integration.py::test_grandqc_artifact_remover_runs_on_real_slide"
    "tests/mussel/utils/test_segmentation_integration.py::test_grandqc_artifact_remover_integrated_with_segment_tissue"
)

EXTRAS=(
    # torch-gpu (tasks 0-29)
    torch-gpu torch-gpu torch-gpu torch-gpu torch-gpu
    torch-gpu torch-gpu torch-gpu torch-gpu torch-gpu
    torch-gpu torch-gpu torch-gpu torch-gpu torch-gpu
    torch-gpu torch-gpu torch-gpu torch-gpu torch-gpu
    torch-gpu torch-gpu torch-gpu torch-gpu torch-gpu
    torch-gpu torch-gpu torch-gpu torch-gpu torch-gpu
    # fastattn (tasks 30-32)
    fastattn fastattn fastattn
    # tensorflow (task 33)
    tensorflow-gpu
    # neural seg + artifact removal (tasks 34-37)
    torch-gpu torch-gpu torch-gpu torch-gpu
)

VENVS=(
    # torch-gpu (tasks 0-29): use repo .venv (empty = no override)
    "" "" "" "" ""  "" "" "" "" ""
    "" "" "" "" ""  "" "" "" "" ""
    "" "" "" "" ""  "" "" "" "" ""
    # fastattn (tasks 30-32)
    "$HOME/venvs/mussel-fastattn"
    "$HOME/venvs/mussel-fastattn"
    "$HOME/venvs/mussel-fastattn"
    # tensorflow (task 33)
    "$HOME/venvs/mussel-tensorflow"
    # neural seg + artifact removal (tasks 34-37): use repo .venv
    "" "" "" ""
)

TEST_NODE="${TESTS[$SLURM_ARRAY_TASK_ID]}"
EXTRA="${EXTRAS[$SLURM_ARRAY_TASK_ID]}"
VENV="${VENVS[$SLURM_ARRAY_TASK_ID]}"
LABEL=$(echo "$TEST_NODE" | sed 's/.*:://;s/\[/_/;s/\]//')

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"

echo "=== mussel integration test: ${LABEL} ==="
echo "Repo:   $REPO_DIR"
echo "Node:   $(hostname)"
echo "GPUs:   ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Python: $(python3 --version 2>&1)"
echo "Extra:  $EXTRA"
echo "Date:   $(date)"
echo ""

if [[ -f ~/.hf_cred.env ]]; then
    source ~/.hf_cred.env
fi

if [[ -n "$VENV" ]]; then
    export UV_PROJECT_ENVIRONMENT="$VENV"
fi

uv sync --extra "$EXTRA"

echo ""
echo "--- Running: ${TEST_NODE} ---"
uv run pytest "$TEST_NODE" \
    --use-gpu \
    -v \
    --tb=short \
    --timeout=600

EXIT_CODE=$?
echo ""
echo "=== Test finished with exit code ${EXIT_CODE} ==="
exit $EXIT_CODE
