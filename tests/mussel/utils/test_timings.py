"""Opt-in stage timings (mussel.utils.timings, timings_path= on the CLIs)."""
import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import mussel.cli.extract_features as extract_cli
import mussel.cli.tessellate as tessellate_cli
from mussel.cli.tessellate import SegConfig, TessellateConfig
from mussel.models import ModelType
from mussel.utils import feature_extract as fe
from mussel.utils import timings

_BATCH = 7


def _records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_not_recording_is_a_no_op():
    items = [1, 2, 3]
    assert not timings.enabled()
    assert timings.timed_iter(items) is items
    with timings.stage("x") as record:
        timings.add(n=1)
        with timings.counting("y_s"):
            pass
    assert record is None


def test_stages_counters_nesting_and_errors(tmp_path):
    out = tmp_path / "t.jsonl"
    with timings.recording(str(out)):
        with timings.stage("outer", slide="a") as outer:
            for _ in timings.timed_iter(range(3)):
                timings.add(batches=1)
                with timings.counting("model_s"):
                    pass
            with timings.stage("inner"):
                timings.add(batches=10)  # counts toward the innermost stage only
        with pytest.raises(ValueError):
            with timings.stage("failing"):
                raise ValueError("boom")
    assert not timings.enabled()
    inner, outer_rec, failing = _records(out)  # written as each stage ends
    assert outer_rec["stage"] == "outer" and outer_rec["slide"] == "a" and outer_rec["batches"] == 3
    assert outer_rec["load_wait_s"] >= 0 and outer_rec["model_s"] >= 0
    assert inner["batches"] == 10 and outer_rec["start"] <= inner["start"] <= inner["end"] <= outer_rec["end"]
    assert outer_rec["seconds"] == pytest.approx(outer_rec["end"] - outer_rec["start"], abs=1e-3)
    assert failing["error"] == "ValueError"
    assert outer is not None


def test_recording_appends(tmp_path):
    out = tmp_path / "t.jsonl"
    for _ in range(2):
        with timings.recording(str(out)), timings.stage("s"):
            pass
    assert len(_records(out)) == 2


def _slides(fakes, tmp_path):
    return [fakes.write_tiles(tmp_path / "a.patch.h5", fakes.coords[:50]),
            fakes.write_tiles(tmp_path / "b.patch.h5", fakes.coords[::-1][:23])], [50, 23]


def _check_patch_records(records, model, n_tiles):
    loads = [r for r in records if r["stage"] == "model_load"]
    slides = [r for r in records if r["stage"] == "patch_features"]
    assert [r["model"] for r in loads] == model.split(",")
    assert [r["tiles"] for r in slides] == n_tiles
    for r, n in zip(slides, n_tiles):  # tiles: in the slide; tiles_done: through the model
        assert r["model"] == model and r["batches"] == math.ceil(n / _BATCH) and r["tiles_done"] == n
        for key in ("load_wait_s", "model_s", "write_s"):
            assert r[key] >= 0
        assert r["model_s"] + r["write_s"] + r["load_wait_s"] <= r["seconds"] + 1e-3


@pytest.mark.parametrize("shared_loader, num_workers", [(True, 2), (False, 2), (False, 0)])
def test_patch_features_batch_records_each_slide(fakes, tmp_path, shared_loader, num_workers):
    h5s, n_tiles = _slides(fakes, tmp_path)
    plain = [str(tmp_path / f"plain_{i}.h5") for i in range(2)]
    timed = [str(tmp_path / f"timed_{i}.h5") for i in range(2)]
    out = tmp_path / "t.jsonl"
    common = dict(model_type=ModelType.HOPTIMUS1, batch_size=_BATCH, use_gpu=False, num_workers=num_workers,
                  pin_memory=False, tile_size_check="off", shared_loader=shared_loader)
    with patch.object(fe, "get_model_factory", side_effect=fakes.factories()):
        fe.extract_patch_features_batch(h5s, [fakes.slide] * 2, plain, **common)
        with timings.recording(str(out)):
            fe.extract_patch_features_batch(h5s, [fakes.slide] * 2, timed, **common)
    _check_patch_records(_records(out), "HOPTIMUS1", n_tiles)
    for a, b in zip(plain, timed):  # recording doesn't change the output
        with h5py.File(a) as fa, h5py.File(b) as fb:
            np.testing.assert_array_equal(fa["features"][:], fb["features"][:])


def test_multi_model_pass_records_each_slide(fakes, tmp_path):
    h5s, n_tiles = _slides(fakes, tmp_path)
    models = [ModelType.HOPTIMUS1, ModelType.OPTIMUS]
    outs = {m: [str(tmp_path / f"{m.code}_{i}.h5") for i in range(2)] for m in models}
    out = tmp_path / "t.jsonl"
    with patch.object(fe, "get_model_factory", side_effect=fakes.factories()), timings.recording(str(out)):
        fe.extract_patch_features_batch_multi(h5s, [fakes.slide] * 2, outs, batch_size=_BATCH, use_gpu=False,
                                              num_workers=2, pin_memory=False, tile_size_check="off")
    _check_patch_records(_records(out), "HOPTIMUS1,OPTIMUS", n_tiles)


def test_slide_encoder_records_each_slide(fakes, tmp_path):
    h5s, n_tiles = _slides(fakes, tmp_path)
    feats = []
    for i, h5 in enumerate(h5s):
        p = tmp_path / f"f{i}.h5"
        with h5py.File(h5) as src, h5py.File(p, "w") as f:
            ds = f.create_dataset("coords", data=src["coords"][:])
            for k, v in src["coords"].attrs.items():
                ds.attrs[k] = v
            f.create_dataset("features", data=np.ones((len(src["coords"]), 4), dtype=np.float32))
        feats.append(str(p))
    model = MagicMock()
    model.get_model_fun.return_value = lambda f, c, s: f.mean(1)
    factory = MagicMock()
    factory.get_model.return_value = model
    out = tmp_path / "t.jsonl"
    with patch.object(fe, "get_model_factory", return_value=factory), timings.recording(str(out)):
        fe.aggregate_slide_features_batch(feats, output_pt_paths=[str(tmp_path / f"{i}.pt") for i in range(2)],
                                          aggregation_method="model", model_type=ModelType.TITAN_SLIDE,
                                          use_gpu=False, tile_size_check="off")
    records = _records(out)
    assert [r["stage"] for r in records] == ["model_load", "slide_encode", "slide_encode"]
    assert [r["tiles"] for r in records[1:]] == n_tiles
    assert all(r["model"] == "TITAN_SLIDE" for r in records)


def test_extract_features_cli_timings_path(fakes, tmp_path):
    h5s, n_tiles = _slides(fakes, tmp_path)
    out = tmp_path / "t.jsonl"
    cfg = OmegaConf.structured(extract_cli.ExtractFeaturesConfig(
        patch_h5_paths=h5s, slide_paths=[fakes.slide] * 2, slide_ids=["A", "B"], output_dir=str(tmp_path / "out"),
        model_type=ModelType.HOPTIMUS1, batch_size=_BATCH, use_gpu=False, num_workers=0,
        tile_size_check="off", timings_path=str(out)))
    with patch.object(fe, "get_model_factory", side_effect=fakes.factories()):
        extract_cli.main(cfg)
    _check_patch_records(_records(out), "HOPTIMUS1", n_tiles)


def test_tessellate_cli_timings_path(tmp_path):
    out = tmp_path / "t.jsonl"
    cfg = TessellateConfig(slide_paths=["a.svs", "b.svs"], slide_ids=["A", "B"], output_dir=str(tmp_path / "tiles"),
                           seg_config=SegConfig(segment_threshold=0, patch_sizes=[224, 256], patch_mpps=[0.5, 1.0]),
                           timings_path=str(out))

    def fake_multi(*, patch_sizes, patch_mpps, output_h5_paths, **kwargs):
        # Like segment_tissue_multi with patch_mpps: keyed by (size, mpp).
        for p in output_h5_paths:
            Path(p).write_text("patch h5")
        return {(s, float(m)): (MagicMock(), MagicMock(), np.zeros((n, 2)))
                for s, m, n in zip(patch_sizes, patch_mpps, (9, 2))}

    with patch.object(tessellate_cli, "segment_tissue_multi", side_effect=fake_multi):
        tessellate_cli.main(OmegaConf.create(cfg))
    records = _records(out)
    assert [r["stage"] for r in records] == ["model_load", "tessellate", "tessellate"]
    assert [r["slide"] for r in records[1:]] == ["a.svs", "b.svs"]
    assert records[1]["tile_sets"] == [{"patch_size": 224, "mpp": 0.5}, {"patch_size": 256, "mpp": 1.0}]
    assert records[1]["tiles"] == {"224px_0.5mpp": 9, "256px_1mpp": 2}


class _LazySegmenter:
    """NeuralTissueSegmenter's lazy-load contract: weights load on first use."""

    def __init__(self):
        self.loaded_at = None

    def load(self):
        if self.loaded_at is None:
            import time
            time.sleep(0.05)
            self.loaded_at = time.time()


def _fake_multi(*, patch_sizes, patch_mpps, output_h5_paths, **kwargs):
    for p in output_h5_paths:
        Path(p).write_text("patch h5")
    return {(s, float(m)): (MagicMock(), MagicMock(), np.zeros((3, 2))) for s, m in zip(patch_sizes, patch_mpps)}


@pytest.mark.parametrize("record", [True, False])
def test_segmentation_model_load_is_timed_in_model_load(tmp_path, record):
    seg = _LazySegmenter()
    out = tmp_path / "t.jsonl"
    cfg = TessellateConfig(slide_paths=["a.svs"], slide_ids=["A"], output_dir=str(tmp_path / "tiles"),
                           seg_config=SegConfig(segment_threshold=0), timings_path=str(out) if record else None)
    with patch.object(tessellate_cli, "_build_neural_segmenter", return_value=seg), \
            patch.object(tessellate_cli, "segment_tissue_multi", side_effect=_fake_multi):
        tessellate_cli.main(OmegaConf.create(cfg))
    if not record:
        assert seg.loaded_at is None  # stays lazy without timings
        return
    load = next(r for r in _records(out) if r["stage"] == "model_load")
    assert load["start"] <= seg.loaded_at <= load["end"] and load["seconds"] >= 0.05


def test_one_step_records_segmentation_load_and_wraps_main(tmp_path):
    import mussel.cli.tessellate_extract_features as tef

    seg = _LazySegmenter()
    out = tmp_path / "t.jsonl"
    cfg = MagicMock(use_gpu=False, gpu_device_id=None, gpu_device_ids=None,
                    seg_config=OmegaConf.create({"seg_model": "neural"}), neural_config=tef.NeuralSegConfig())
    with patch.object(tef, "_build_neural_segmenter", return_value=seg), \
            patch.object(tef, "_build_artifact_remover", return_value=None), timings.recording(str(out)):
        tef._build_segmentation_runtime(cfg)
    (load,) = _records(out)
    assert load["stage"] == "model_load" and load["start"] <= seg.loaded_at <= load["end"]

    seen = {}
    with patch.object(tef, "_main", side_effect=lambda c: seen.setdefault("enabled", timings.enabled())):
        tef.main(OmegaConf.create({"timings_path": str(tmp_path / "m.jsonl")}))
    assert seen["enabled"] is True and not timings.enabled()


def test_failed_slide_encoding_is_recorded_and_the_batch_continues(fakes, tmp_path):
    h5s, n_tiles = _slides(fakes, tmp_path)
    feats = []
    for i, h5 in enumerate(h5s):
        p = tmp_path / f"f{i}.h5"
        with h5py.File(h5) as src, h5py.File(p, "w") as f:
            ds = f.create_dataset("coords", data=src["coords"][:])
            for k, v in src["coords"].attrs.items():
                ds.attrs[k] = v
            f.create_dataset("features", data=np.ones((len(src["coords"]), 4), dtype=np.float32))
        feats.append(str(p))

    def encode(f, c, s):
        if f.shape[1] == n_tiles[1]:
            raise RuntimeError("CUDA out of memory")
        return f.mean(1)

    model = MagicMock()
    model.get_model_fun.return_value = encode
    factory = MagicMock()
    factory.get_model.return_value = model
    out = tmp_path / "t.jsonl"
    pts = [str(tmp_path / f"{i}.pt") for i in range(2)]
    with patch.object(fe, "get_model_factory", return_value=factory), timings.recording(str(out)):
        # The batch goes on past the failed slide, then reports it.
        with pytest.raises(fe.SlideBatchError, match="1 of 2 slide"):
            fe.aggregate_slide_features_batch(feats, output_pt_paths=pts, aggregation_method="model",
                                              model_type=ModelType.TITAN_SLIDE, use_gpu=False,
                                              tile_size_check="off")
    enc = [r for r in _records(out) if r["stage"] == "slide_encode"]
    assert [r.get("error") for r in enc] == [None, "RuntimeError"]
    assert Path(pts[0]).exists() and not Path(pts[1]).exists()


def test_gpu_peak_is_reset_for_outermost_stages_only(tmp_path):
    class FakeCuda:
        def __init__(self):
            self.resets, self.peak = 0, 0

        def reset_peak_memory_stats(self):
            self.resets += 1
            self.peak = 0

        def max_memory_allocated(self):
            return self.peak

    cuda = FakeCuda()
    out = tmp_path / "t.jsonl"
    with patch.object(timings, "_cuda", return_value=cuda), timings.recording(str(out)):
        with timings.stage("outer"):
            cuda.peak = 3 * 2**20
            with timings.stage("inner"):
                cuda.peak = 5 * 2**20
        with timings.stage("next"):
            cuda.peak = 2 * 2**20
    inner, outer, nxt = _records(out)
    assert cuda.resets == 2  # outer and next; not inner
    assert (inner["gpu_mem_peak_mib"], outer["gpu_mem_peak_mib"], nxt["gpu_mem_peak_mib"]) == (5.0, 5.0, 2.0)
