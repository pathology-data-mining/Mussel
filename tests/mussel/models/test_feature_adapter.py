"""FeatureAdapter: rebuilds ridge / MLP adapters from their checkpoints and applies them."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mussel.models.feature_adapter import FeatureAdapter


def _save(path, kind, net, in_dim):
    torch.manual_seed(1)
    mean, std = torch.randn(in_dim), torch.rand(in_dim) + 0.5
    sd = {
        "mean": mean,
        "std": std,
        **{f"net.{k}": v for k, v in net.state_dict().items()},
    }
    torch.save(
        {"state_dict": sd, "kind": kind, "teacher": "keep", "in_dim": in_dim}, path
    )
    return lambda x: F.normalize(net((x - mean) / std), dim=-1)


@pytest.mark.parametrize("kind", ["ridge", "mlp"])
def test_load_and_apply(tmp_path, kind):
    torch.manual_seed(0)
    net = (
        nn.Linear(12, 6)
        if kind == "ridge"
        else nn.Sequential(nn.Linear(12, 20), nn.GELU(), nn.Linear(20, 6))
    )
    ref = _save(tmp_path / "a.pt", kind, net, 12)
    adapter = FeatureAdapter.load(str(tmp_path / "a.pt"))
    x = torch.randn(50, 12)
    out = adapter.apply(
        x.to(torch.bfloat16).float(), batch_size=7
    )  # stored features are bfloat16
    with torch.no_grad():
        torch.testing.assert_close(out, ref(x.to(torch.bfloat16).float()))
    assert adapter.teacher == "keep" and adapter.in_dim == 12 and out.shape == (50, 6)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(50))


def test_wrong_feature_dim_is_rejected(tmp_path):
    _save(tmp_path / "a.pt", "ridge", nn.Linear(12, 6), 12)
    with pytest.raises(ValueError, match="adapter expects 12"):
        FeatureAdapter.load(str(tmp_path / "a.pt")).apply(torch.randn(3, 10))


def test_unknown_kind_is_rejected(tmp_path):
    _save(tmp_path / "a.pt", "transformer", nn.Linear(12, 6), 12)
    with pytest.raises(ValueError, match="unknown adapter kind"):
        FeatureAdapter.load(str(tmp_path / "a.pt"))
