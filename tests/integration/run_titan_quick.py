"""Quick TITAN integration test — fits on any GPU (N=5k)."""
import sys
sys.path.insert(0, "/gpfs/mskmind_ess/limr/repos/Mussel-titan-fix")
sys.path.insert(1, "/gpfs/mskmind_ess/limr/repos/Mussel")

import torch, importlib.util

spec = importlib.util.spec_from_file_location(
    "mussel.models.conch",
    "/gpfs/mskmind_ess/limr/repos/Mussel-titan-fix/mussel/models/conch.py",
)
conch_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conch_mod)
sys.modules["mussel.models.conch"] = conch_mod

from mussel.models.model_factory import ModelType, get_model_factory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

model = get_model_factory(ModelType.TITAN_SLIDE).get_model(
    ModelType.TITAN_SLIDE.path, use_gpu=(DEVICE=="cuda")
)
model_fun = model.get_model_fun()

PASSED, FAILED = [], []
def test(name, fn):
    try: fn(); print(f"  PASS: {name}"); PASSED.append(name)
    except Exception as e: print(f"  FAIL: {name}: {e}"); FAILED.append(name)

def t_shape():
    N, D, ps = 1000, 768, 420
    f = torch.randn(1, N, D); c = torch.zeros(1, N, 2, dtype=torch.int64)
    r = model_fun(f, c, ps)
    assert r.shape == (768,), f"shape={r.shape}"
    assert torch.isfinite(r).all()

def t_moderate_n():
    """N=5k — validates patches work end-to-end on any GPU."""
    N, D, ps = 5000, 768, 420
    W = int(N**0.5)+1; H = (N+W-1)//W
    coords = torch.stack(torch.meshgrid(torch.arange(W)*ps, torch.arange(H)*ps, indexing='ij'),
                         dim=-1).reshape(-1,2)[:N].unsqueeze(0).to(torch.int64)
    f = torch.randn(1, N, D)
    if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats(0)
    r = model_fun(f, coords, ps)
    if DEVICE == "cuda":
        peak = torch.cuda.max_memory_allocated(0)/1e9
        print(f"    VRAM peak: {peak:.1f} GB")
    assert r.shape == (768,) and torch.isfinite(r).all()

def t_cpu_ram():
    import resource
    N, D, ps = 10000, 768, 420
    W = int(N**0.5)+1; H = (N+W-1)//W
    coords = torch.stack(torch.meshgrid(torch.arange(W)*ps, torch.arange(H)*ps, indexing='ij'),
                         dim=-1).reshape(-1,2)[:N].unsqueeze(0).to(torch.int64)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    r = model_fun(torch.randn(1,N,D), coords, ps)
    delta = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before)/1e6
    print(f"    CPU RAM delta: {delta:.2f} GB")
    assert delta < 5.0 and r.shape == (768,) and torch.isfinite(r).all()

print("\n=== Running ===")
test("Small N (1k)", t_shape)
test("Moderate N (5k) — end-to-end patches validated", t_moderate_n)
test("CPU RAM bounded (N=10k)", t_cpu_ram)
print(f"\n=== {len(PASSED)}/3 passed ===")
sys.exit(0 if not FAILED else 1)
