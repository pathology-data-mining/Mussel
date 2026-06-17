#!/bin/bash
#SBATCH --job-name=conch-fa2-regression
#SBATCH --partition=hpc
#SBATCH --qos=premium
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pllimsksparky[1-4]
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/gpfs/mskmind_ess/limr/repos/Mussel-titan-fix/tests/integration/conch-fa2-regression-%j.log
#SBATCH --export=NONE

set -euo pipefail

REPO=/gpfs/mskmind_ess/limr/repos/Mussel-titan-fix
SIF=/gpfs/mskmind_ess/limr/repos/mussel-nf/mussel-fastattn.sif
CACHE=/gpfs/cdsi_ess/home/limr/.cache

echo "=== CONCH1.5 flash_attention_2 regression test ==="
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
echo "Date: $(date)"

# Run under the fastattn SIF so flash_attention_2 is available
apptainer exec --nv \
  --bind ${REPO}:/repo \
  --bind ${CACHE}:/root/.cache \
  ${SIF} \
  python -c "
from mussel.models.base import get_best_attn_implementation
impl = get_best_attn_implementation()
print(f'Attention impl: {impl}')
if impl != 'flash_attention_2':
    raise RuntimeError(f'Expected flash_attention_2 on A100, got: {impl}')
print('flash_attention_2 confirmed')
"

apptainer exec --nv \
  --bind ${REPO}:/repo \
  --bind ${CACHE}:/root/.cache \
  ${SIF} \
  python -m pytest /repo/tests/mussel/models/test_encoder_integration.py::test_patch_encoder_matches_snapshot \
    -k CONCH1_5 --use-gpu -v --tb=short \
    -p no:cacheprovider \
    --override-ini="addopts=-v --tb=short"

echo "=== Done ==="
