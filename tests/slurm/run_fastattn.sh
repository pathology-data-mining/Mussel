#!/bin/bash
# SLURM job: integration tests for fastattn-based models (Prov-GigaPath).
#
# Submit from the repo root:
#   sbatch tests/slurm/run_fastattn.sh
#
# GLIBC compatibility strategy (RHEL 8, GLIBC 2.28):
#   The pre-built flash_attn_2_cuda.so requires GLIBC_2.32 (__libc_single_threaded)
#   and GLIBCXX_3.4.29 / CXXABI_1.3.13 (from libstdc++).  We satisfy these via:
#     - LD_PRELOAD: ~/libcompat/libglibc_compat.so  (provides __libc_single_threaded@GLIBC_2.32)
#     - LD_LIBRARY_PATH: NVIDIA Nsight's bundled libstdc++ (has GLIBCXX_3.4.30, CXXABI_1.3.13)
#   The compat stub is built on first run and cached at ~/libcompat/.
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

# ---------------------------------------------------------------------------
# GLIBC / libstdc++ compatibility shims
#
# The pre-built flash_attn_2_cuda.so requires:
#   - __libc_single_threaded@GLIBC_2.32  (in libc.so.6 >= 2.32)
#   - GLIBCXX_3.4.29 / CXXABI_1.3.13    (in libstdc++.so.6 from GCC 11+)
#
# On RHEL 8 (GLIBC 2.28, GCC 8.5) we provide these as follows:
#   1. Build a tiny stub library that exports __libc_single_threaded@GLIBC_2.32
#      (safe to set to 0 = multi-threaded; only disables a micro-optimisation)
#   2. Prepend NVIDIA Nsight's newer libstdc++ (ships with CUDA toolkit) via
#      LD_LIBRARY_PATH so the dynamic linker finds GLIBCXX_3.4.29+
# ---------------------------------------------------------------------------
COMPAT_DIR="$HOME/libcompat"
COMPAT_SO="$COMPAT_DIR/libglibc_compat.so"
NSIGHT_LIBDIR="/opt/nvidia/nsight-systems/2024.4.2/host-linux-x64"

if [[ ! -f "$COMPAT_SO" ]]; then
    echo ""
    echo "--- Building GLIBC compat stub (one-time) ---"
    mkdir -p "$COMPAT_DIR"

    cat > "$COMPAT_DIR/glibc_compat.c" << 'CEOF'
/* Provides __libc_single_threaded@GLIBC_2.32 on GLIBC < 2.32 nodes.
   Value 0 (multi-threaded) is conservative and always safe. */
volatile int __libc_single_threaded = 0;
CEOF

    cat > "$COMPAT_DIR/glibc_compat.map" << 'MEOF'
GLIBC_2.32 {
    global:
        __libc_single_threaded;
};
MEOF

    gcc -shared -fPIC -o "$COMPAT_SO" \
        "$COMPAT_DIR/glibc_compat.c" \
        -Wl,--version-script="$COMPAT_DIR/glibc_compat.map"
    echo "    Built: $COMPAT_SO"
fi

# Apply shims for this session
export LD_PRELOAD="$COMPAT_SO"
if [[ -d "$NSIGHT_LIBDIR" ]]; then
    export LD_LIBRARY_PATH="$NSIGHT_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "    Using Nsight libstdc++ from: $NSIGHT_LIBDIR"
fi

# Verify flash_attn loads before running tests
echo ""
echo "--- Verifying flash_attn loads ---"
"${UV_PROJECT_ENVIRONMENT}/bin/python" -c "import flash_attn; print('flash_attn', flash_attn.__version__)"

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
