#!/bin/bash
#SBATCH --job-name=conch-fa2-regen
#SBATCH --partition=hpc
#SBATCH --qos=premium
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pllimsksparky[1-4]
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/gpfs/mskmind_ess/limr/repos/Mussel-titan-fix/tests/integration/conch-fa2-regen-%j.log
#SBATCH --export=NONE

set -euo pipefail

REPO=/gpfs/mskmind_ess/limr/repos/Mussel-titan-fix
SIF=/gpfs/mskmind_ess/limr/repos/mussel-nf/mussel-fastattn.sif
CACHE=/gpfs/cdsi_ess/home/limr/.cache

echo "=== CONCH1.5 FA2 snapshot regen + verify ==="
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
echo "Date: $(date)"

# Confirm FA2 is active
apptainer exec --nv \
  --bind ${REPO}:/repo \
  --bind ${CACHE}:/root/.cache \
  ${SIF} \
  python -c "
from mussel.models.base import get_best_attn_implementation
impl = get_best_attn_implementation()
print(f'Attention impl: {impl}')
assert impl == 'flash_attention_2', f'Expected flash_attention_2, got: {impl}'
"

# Step 1: regenerate snapshot with FA2
echo "--- Step 1: regenerate snapshot ---"
apptainer exec --nv \
  --bind ${REPO}:/repo \
  --bind ${CACHE}:/root/.cache \
  ${SIF} \
  python -m pytest /repo/tests/mussel/models/test_encoder_integration.py::test_patch_encoder_matches_snapshot \
    -k CONCH1_5 --use-gpu --update-snapshots -v --tb=short \
    -p no:cacheprovider \
    --override-ini="addopts=-v --tb=short"

echo "--- Step 2: verify snapshot ---"
apptainer exec --nv \
  --bind ${REPO}:/repo \
  --bind ${CACHE}:/root/.cache \
  ${SIF} \
  python -m pytest /repo/tests/mussel/models/test_encoder_integration.py::test_patch_encoder_matches_snapshot \
    -k CONCH1_5 --use-gpu -v --tb=short \
    -p no:cacheprovider \
    --override-ini="addopts=-v --tb=short"

echo "=== Done ==="
