"""Tests for importing MUSK's model code (mussel.models.musk._import_musk_modeling).

A fake ``musk`` package stands in for the real one, so these run without MUSK,
a GPU, or its weights.  The fake mirrors MUSK's import chain:
``musk.modeling`` imports ``musk.torchscale.component.flash_attention``.
"""

import sys

import pytest

from mussel.models.musk import (_MUSK_FLASH_ATTENTION_MODULE,
                                _import_musk_modeling)

_BROKEN_BACKEND = (
    "raise ImportError("
    "'Requires Flash-Attention version >=2.7.1,<=2.8.4 but got 2.6.3.')\n"
)
_WORKING_BACKEND = "flash_attn_func = 'real-backend'\n"


@pytest.fixture
def fake_musk(tmp_path, monkeypatch):
    """Create a fake ``musk`` package; return a function that sets its backend."""
    component = tmp_path / "musk" / "torchscale" / "component"
    component.mkdir(parents=True)
    for pkg in (tmp_path / "musk", tmp_path / "musk" / "torchscale", component):
        (pkg / "__init__.py").write_text("")
    (tmp_path / "musk" / "modeling.py").write_text(
        "from musk.torchscale.component.flash_attention import flash_attn_func\n"
        "BACKEND = flash_attn_func\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    def _drop_musk_modules():
        for name in [m for m in sys.modules if m == "musk" or m.startswith("musk.")]:
            del sys.modules[name]

    _drop_musk_modules()

    def set_backend(source: str) -> None:
        (component / "flash_attention.py").write_text(source)

    yield set_backend
    _drop_musk_modules()


def test_working_backend_is_left_alone(fake_musk):
    fake_musk(_WORKING_BACKEND)
    _import_musk_modeling()
    assert sys.modules["musk.modeling"].BACKEND == "real-backend"


def test_unimportable_backend_falls_back_to_musk_default(fake_musk, caplog):
    fake_musk(_BROKEN_BACKEND)
    _import_musk_modeling()
    assert sys.modules["musk.modeling"].BACKEND is None
    assert sys.modules[_MUSK_FLASH_ATTENTION_MODULE].flash_attn_func is None
    assert "fused-attention backend failed to import" in caplog.text


class _HideMusk:
    """Meta-path finder that makes ``musk`` look uninstalled."""

    def find_spec(self, name, path=None, target=None):
        if name == "musk" or name.startswith("musk."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


def test_missing_musk_keeps_module_not_found_text(monkeypatch):
    for name in [m for m in sys.modules if m == "musk" or m.startswith("musk.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_HideMusk(), *sys.meta_path])
    with pytest.raises(ImportError) as excinfo:
        _import_musk_modeling()
    message = str(excinfo.value).lower()
    assert "no module named 'musk" in message
    assert "pip install git+https://github.com/lilab-stanford/musk" in message
