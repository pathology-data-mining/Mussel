#!/usr/bin/env python3
"""GPU integration test for TITAN get_alibi monkey-patch.

Verifies:
1. No OOM for N=30k patches on A100 (tests the actual fix)
2. Output shape (768,) and finite values
3. GPU peak VRAM stays within bounds

Run via SLURM:
  sbatch --qos=premium --gpus=1 --mem=64G --time=0:30:00 \
    --output=test_titan_gpu.log \
    /gpfs/mskmind_ess/limr/repos/Mussel-titan-fix/tests/integration/test_titan_gpu_integration.py
"""
import sys
sys.path.insert(0, "/gpfs/mskmind_ess/limr/repos/Mussel-titan-fix")
sys.path.insert(1, "/gpfs/mskmind_ess/limr/repos/Mussel")

import torch
import importlib.util

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PASSED = []
FAILED = []


def test(name, fn):
    try:
        fn()
        print(f"  PASS: {name}")
        PASSED.append(name)
    except Exception as e:
        import traceback
        print(f"  FAIL: {name}: {e}")
        traceback.print_exc()
        FAILED.append(name)


# ---------------------------------------------------------------------------
# Load patched model (from worktree, not prod)
# ---------------------------------------------------------------------------
print("\n=== Loading patched TitanSlideEncoderModel from worktree ===")

spec = importlib.util.spec_from_file_location(
    "mussel.models.conch",
    "/gpfs/mskmind_ess/limr/repos/Mussel-titan-fix/mussel/models/conch.py",
)
conch_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conch_mod)
sys.modules["mussel.models.conch"] = conch_mod

from mussel.models.model_factory import ModelType, get_model_factory

model_factory = get_model_factory(ModelType.TITAN_SLIDE)
model = model_factory.get_model(ModelType.TITAN_SLIDE.path, use_gpu=(DEVICE == "cuda"))
model_fun = model.get_model_fun()
print(f"  Model loaded on device: {model.device}")

# TITAN's mlp_patch_embed_dim=768, so CONCH output features must be 768-dim
CONCH_DIM = 768
PATCH_SIZE = 420  # IMPACT: 224px patch at 0.5 MPP, level-0 size = 420px


def _compact_grid_coords(N: int, step: int = 420):
    """Create a compact 2D rectangular grid of N patch coordinates.
    Avoids diagonal layouts which create N×N-cell bounding boxes.
    """
    W = int(N ** 0.5) + 1
    H = (N + W - 1) // W
    coords = torch.stack(torch.meshgrid(
        torch.arange(W) * step,
        torch.arange(H) * step,
        indexing='ij',
    ), dim=-1).reshape(-1, 2)[:N]
    return coords.unsqueeze(0).to(torch.int64)  # (1, N, 2)


# ---------------------------------------------------------------------------
# Test 1: Small N (1k patches) — shape + finite values
# ---------------------------------------------------------------------------
def t1_small_n():
    N = 1000
    features = torch.randn(1, N, CONCH_DIM, dtype=torch.float32)
    coords = _compact_grid_coords(N)
    result = model_fun(features, coords, PATCH_SIZE)
    assert result.shape == (768,), f"Shape: {result.shape}"
    assert torch.isfinite(result).all(), "Non-finite values"
    assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 2: Large N (30k patches) — no GPU OOM
# ---------------------------------------------------------------------------
def t2_large_n_no_oom():
    if DEVICE == "cpu":
        print("    (skipping — GPU required)")
        return

    N = 30_000
    torch.cuda.reset_peak_memory_stats(0)

    features = torch.randn(1, N, CONCH_DIM, dtype=torch.float32)
    grid = _compact_grid_coords(N)  # compact ~173×174 grid

    result = model_fun(features, grid, PATCH_SIZE)

    vram_peak = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"    GPU VRAM peak: {vram_peak:.1f} GB")

    assert result.shape == (768,), f"Shape: {result.shape}"
    assert torch.isfinite(result).all(), "Non-finite values"
    # V100=16 GB: 30k patches → all_bias ~2.6 GB + model → should fit
    # A100=80 GB: plenty of headroom
    assert vram_peak < (16.0 if torch.cuda.get_device_properties(0).total_memory < 20e9 else 70.0), \
        f"VRAM peak {vram_peak:.1f} GB too high"


# ---------------------------------------------------------------------------
# Test 3: CPU RAM bounded (no numpy OOM for N=10k)
# ---------------------------------------------------------------------------
def t3_cpu_ram_bounded():
    import resource
    N = 10_000
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux

    features = torch.randn(1, N, CONCH_DIM)
    coords = _compact_grid_coords(N)
    result = model_fun(features, coords, PATCH_SIZE)

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    delta_gb = (rss_after - rss_before) / 1e6
    print(f"    CPU RAM delta: {delta_gb:.2f} GB")
    # Original numpy would need ~7 GB for N=10k; patched should be <2 GB
    assert delta_gb < 5.0, f"CPU RAM delta {delta_gb:.2f} GB (expected <5 GB)"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print("\n=== Running tests ===")
test("Small N (1k patches) — shape + finite", t1_small_n)
test("Large N (30k patches) — no GPU OOM", t2_large_n_no_oom)
test("CPU RAM bounded (N=10k)", t3_cpu_ram_bounded)

print(f"\n=== Results: {len(PASSED)} passed, {len(FAILED)} failed ===")
if FAILED:
    sys.exit(1)
print("ALL TESTS PASSED")
