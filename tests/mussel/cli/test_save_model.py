import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from mussel.cli.save_model import (
    SaveModelConfig,
    _ensure_cache_dirs,
    _save_one_model,
    save_model,
)
from mussel.models import ModelType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(save_side_effect=None):
    model = MagicMock()
    if save_side_effect is not None:
        model.save.side_effect = save_side_effect
    return model


def _make_mock_factory(model):
    factory = MagicMock()
    factory.get_model.return_value = model
    return factory


# ---------------------------------------------------------------------------
# _ensure_cache_dirs
# ---------------------------------------------------------------------------

class TestEnsureCacheDirs:
    def test_creates_standard_dirs(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        _ensure_cache_dirs()

        assert (fake_home / ".cache").is_dir()
        assert (fake_home / ".cache" / "huggingface").is_dir()
        assert (fake_home / ".cache" / "torch").is_dir()

    def test_skips_when_hf_home_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "custom_hf"))
        monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        _ensure_cache_dirs()

        assert not (fake_home / ".cache").exists()

    def test_skips_when_transformers_cache_set(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "custom_tc"))
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        _ensure_cache_dirs()

        assert not (fake_home / ".cache").exists()

    def test_handles_symlink_to_existing_target(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        # Pre-create .cache as a symlink pointing to a real dir
        real_cache = tmp_path / "real_cache"
        real_cache.mkdir()
        (fake_home / ".cache").symlink_to(real_cache)

        _ensure_cache_dirs()  # should not raise

        assert (fake_home / ".cache").is_symlink()
        assert (real_cache / "huggingface").is_dir()
        assert (real_cache / "torch").is_dir()


# ---------------------------------------------------------------------------
# _save_one_model
# ---------------------------------------------------------------------------

class TestSaveOneModel:
    @patch("mussel.cli.save_model.get_model_factory")
    def test_skips_if_ready_marker_exists(self, mock_gmf, tmp_path):
        model_dir = tmp_path / "RESNET50"
        model_dir.mkdir()
        (model_dir / ".ready").write_text("cached\n")

        _save_one_model(ModelType.RESNET50, tmp_path)

        mock_gmf.assert_not_called()

    @patch("mussel.cli.save_model.get_model_factory")
    def test_skips_if_pth_file_exists(self, mock_gmf, tmp_path):
        pth = tmp_path / "RESNET50.pth"
        pth.write_text("fake")

        _save_one_model(ModelType.RESNET50, tmp_path)

        mock_gmf.assert_not_called()

    @patch("mussel.cli.save_model._ensure_cache_dirs")
    @patch("mussel.cli.save_model.get_model_factory")
    def test_saves_to_directory_on_success(self, mock_gmf, mock_ecd, tmp_path):
        model = _make_mock_model()
        mock_gmf.return_value = _make_mock_factory(model)

        _save_one_model(ModelType.RESNET50, tmp_path)

        model_dir = tmp_path / "RESNET50"
        model.save.assert_called_once_with(str(model_dir))
        assert (model_dir / ".ready").read_text() == "cached\n"
        mock_ecd.assert_called_once()

    @patch("mussel.cli.save_model._ensure_cache_dirs")
    @patch("mussel.cli.save_model.get_model_factory")
    def test_falls_back_to_pth_when_dir_save_raises_not_implemented(
        self, mock_gmf, mock_ecd, tmp_path
    ):
        model = _make_mock_model(
            save_side_effect=[NotImplementedError("not supported"), None]
        )
        mock_gmf.return_value = _make_mock_factory(model)

        _save_one_model(ModelType.RESNET50, tmp_path)

        model_dir = tmp_path / "RESNET50"
        model_file = tmp_path / "RESNET50.pth"
        assert model.save.call_count == 2
        model.save.assert_any_call(str(model_dir))
        model.save.assert_any_call(str(model_file))
        assert not (model_dir / ".ready").exists()

    @patch("mussel.cli.save_model._ensure_cache_dirs")
    @patch("mussel.cli.save_model.get_model_factory")
    def test_falls_back_to_pth_when_dir_save_raises_value_error(
        self, mock_gmf, mock_ecd, tmp_path
    ):
        model = _make_mock_model(
            save_side_effect=[ValueError("bad path"), None]
        )
        mock_gmf.return_value = _make_mock_factory(model)

        _save_one_model(ModelType.RESNET50, tmp_path)

        assert model.save.call_count == 2

    @patch("mussel.cli.save_model._ensure_cache_dirs")
    @patch("mussel.cli.save_model.get_model_factory")
    def test_prints_skip_message_when_both_save_attempts_fail(
        self, mock_gmf, mock_ecd, tmp_path, capsys
    ):
        model = _make_mock_model(save_side_effect=NotImplementedError("nope"))
        mock_gmf.return_value = _make_mock_factory(model)

        _save_one_model(ModelType.RESNET50, tmp_path)

        assert model.save.call_count == 2
        out = capsys.readouterr().out
        assert "cannot be saved locally" in out


# ---------------------------------------------------------------------------
# save_model — single model mode
# ---------------------------------------------------------------------------

class TestSaveModelSingle:
    def test_raises_if_no_model_type(self):
        cfg = SaveModelConfig(output_path="/tmp/out.pth")
        with pytest.raises(ValueError, match="model_type"):
            save_model(cfg)

    def test_raises_if_no_output_path(self):
        cfg = SaveModelConfig(model_type=ModelType.RESNET50)
        with pytest.raises(ValueError, match="output_path"):
            save_model(cfg)

    @patch("mussel.cli.save_model.get_model_factory")
    def test_single_mode_calls_save(self, mock_gmf, tmp_path):
        out = str(tmp_path / "model.pth")
        model = _make_mock_model()
        mock_gmf.return_value = _make_mock_factory(model)

        cfg = SaveModelConfig(model_type=ModelType.RESNET50, output_path=out)
        save_model(cfg)

        mock_gmf.assert_called_once_with(ModelType.RESNET50)
        mock_gmf.return_value.get_model.assert_called_once_with(None, use_gpu=False)
        model.save.assert_called_once_with(out)

    @patch("mussel.cli.save_model.get_model_factory")
    def test_single_mode_passes_model_path(self, mock_gmf, tmp_path):
        out = str(tmp_path / "model.pth")
        custom_path = "/weights/custom.pth"
        model = _make_mock_model()
        mock_gmf.return_value = _make_mock_factory(model)

        cfg = SaveModelConfig(
            model_type=ModelType.RESNET50,
            model_path=custom_path,
            output_path=out,
        )
        save_model(cfg)

        mock_gmf.return_value.get_model.assert_called_once_with(custom_path, use_gpu=False)


# ---------------------------------------------------------------------------
# save_model — multi-model mode
# ---------------------------------------------------------------------------

class TestSaveModelMulti:
    def test_raises_if_no_model_dir(self):
        cfg = SaveModelConfig(model_types=[ModelType.RESNET50])
        with pytest.raises(ValueError, match="model_dir"):
            save_model(cfg)

    @patch("mussel.cli.save_model._save_one_model")
    def test_creates_output_dir(self, mock_som, tmp_path):
        out_dir = tmp_path / "models"
        cfg = SaveModelConfig(
            model_types=[ModelType.RESNET50],
            model_dir=str(out_dir),
        )
        save_model(cfg)

        assert out_dir.is_dir()

    @patch("mussel.cli.save_model._save_one_model")
    def test_calls_save_one_model_for_each_type(self, mock_som, tmp_path):
        cfg = SaveModelConfig(
            model_types=[ModelType.RESNET50, ModelType.UNI],
            model_dir=str(tmp_path),
        )
        save_model(cfg)

        assert mock_som.call_count == 2
        mock_som.assert_any_call(ModelType.RESNET50, tmp_path)
        mock_som.assert_any_call(ModelType.UNI, tmp_path)

    @patch("mussel.cli.save_model._save_one_model")
    def test_conch1_5_skipped_when_titan_slide_present(self, mock_som, tmp_path):
        cfg = SaveModelConfig(
            model_types=[ModelType.CONCH1_5, ModelType.TITAN_SLIDE],
            model_dir=str(tmp_path),
        )
        save_model(cfg)

        # CONCH1_5 is skipped; only TITAN_SLIDE is saved
        calls = [c.args[0] for c in mock_som.call_args_list]
        assert ModelType.CONCH1_5 not in calls
        assert ModelType.TITAN_SLIDE in calls

    @patch("mussel.cli.save_model._save_one_model")
    def test_conch1_5_redirected_to_titan_when_titan_absent(self, mock_som, tmp_path):
        cfg = SaveModelConfig(
            model_types=[ModelType.CONCH1_5],
            model_dir=str(tmp_path),
        )
        save_model(cfg)

        calls = [c.args[0] for c in mock_som.call_args_list]
        assert ModelType.CONCH1_5 not in calls
        assert ModelType.TITAN_SLIDE in calls

    @patch("mussel.cli.save_model._save_one_model")
    def test_reraises_exception_from_save_one_model(self, mock_som, tmp_path):
        mock_som.side_effect = RuntimeError("download failed")
        cfg = SaveModelConfig(
            model_types=[ModelType.RESNET50],
            model_dir=str(tmp_path),
        )
        with pytest.raises(RuntimeError, match="download failed"):
            save_model(cfg)
