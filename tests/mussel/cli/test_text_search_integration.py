"""End to end: text-to-tile search with real KEEP weights on the test slide.

KEEP tile features (extract_patch_features) -> KEEP prompts (create_class_embeddings)
-> annotate with top_k, with and without a feature adapter.

Integration tests (deselected by default; run with ``-m integration``). KEEP's weights
(~1.7 GB) come from ``hf-hub:Astaxanthin/KEEP``, or from the local directory in
``MUSSEL_KEEP_PATH``; the tests skip if they cannot be loaded.
"""

import json
import os

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

import mussel.cli.annotate
import mussel.cli.create_class_embeddings
from mussel.cli.annotate import AnnotateConfig
from mussel.cli.create_class_embeddings import ClassEmbeddingConfig
from mussel.models.model_factory import ModelType
from mussel.utils.feature_extract import extract_patch_features

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_KEEP = os.environ.get("MUSSEL_KEEP_PATH", "hf-hub:Astaxanthin/KEEP")
_PROMPTS = ["invasive carcinoma", "adipose tissue", "tumor-infiltrating lymphocytes"]
_N_TILES = 48
_TESTDATA = os.path.join(os.path.dirname(__file__), "..", "..", "testdata")


def _subset_tiles(test_data_path, path):
    with h5py.File(os.path.join(test_data_path, "948176.patch.h5")) as f:
        coords, attrs = f["coords"][:_N_TILES], dict(f["coords"].attrs)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("coords", data=coords)
        for k, v in attrs.items():
            ds.attrs[k] = v
    return str(path), coords


@pytest.fixture(scope="module")
def keep_run(tmp_path_factory):
    """KEEP tile features and prompt embeddings for the first tiles of the test slide."""
    tmp = tmp_path_factory.mktemp("keep_text_search")
    tiles, coords = _subset_tiles(_TESTDATA, tmp / "tiles.patch.h5")
    slide = os.path.join(_TESTDATA, "948176.svs")
    try:
        extract_patch_features(
            tiles,
            slide,
            str(tmp / "keep.features.h5"),
            model_type=ModelType.KEEP,
            model_path=None if _KEEP.startswith("hf-hub:") else _KEEP,
            batch_size=16,
            use_gpu=torch.cuda.is_available(),
            num_workers=0,
            pin_memory=False,
            tile_size_check="off",
        )
        mussel.cli.create_class_embeddings.main(
            ClassEmbeddingConfig(
                classes=_PROMPTS,
                output_pt_path=str(tmp / "prompts.pt"),
                model_type=ModelType.KEEP,
                model_path=None if _KEEP.startswith("hf-hub:") else _KEEP,
            )
        )
    except Exception as exc:  # weights unavailable (network, cache)
        pytest.skip(f"KEEP weights could not be loaded: {exc}")
    with h5py.File(tmp / "keep.features.h5") as f:
        feats = torch.from_numpy(f["features"][:]).float()
    torch.save(feats, tmp / "keep.features.pt")
    return dict(tmp=tmp, tiles=tiles, coords=coords, slide=slide, feats=feats)


def _annotate(run, out, **kw):
    mussel.cli.annotate.main(
        AnnotateConfig(
            features_pt_path=str(run["tmp"] / "keep.features.pt"),
            class_embedding_pt_path=str(run["tmp"] / "prompts.pt"),
            classes=_PROMPTS,
            output_csv_path=str(out),
            patch_path=run["tiles"],
            top_k=5,
            **kw,
        )
    )
    return pd.read_csv(out), pd.read_csv(str(out).replace(".csv", ".top5.csv"))


def test_keep_prompts_are_normalised_768_dim_with_their_class_list(keep_run):
    emb = torch.load(keep_run["tmp"] / "prompts.pt", weights_only=True)
    assert emb.shape == (len(_PROMPTS), 768)
    torch.testing.assert_close(emb.norm(dim=-1), torch.ones(len(_PROMPTS)))
    side = json.loads((keep_run["tmp"] / "prompts.pt.classes.json").read_text())
    assert side == {"classes": _PROMPTS, "model_type": "KEEP"}
    assert keep_run["feats"].shape == (_N_TILES, 768)


def test_annotate_ranks_real_keep_tiles_against_real_prompts(keep_run, tmp_path):
    scores, top = _annotate(
        keep_run,
        tmp_path / "scores.csv",
        interrogate=True,
        slide_path=keep_run["slide"],
        interrogation_report_path=str(tmp_path / "report.html"),
    )
    assert list(scores.columns) == _PROMPTS and len(scores) == _N_TILES
    assert scores.abs().le(1.0).all().all()
    for name, rows in top.groupby("class"):
        rows = rows.sort_values("rank")
        assert rows["tile_index"].tolist() == scores[name].nlargest(5).index.tolist()
        np.testing.assert_allclose(
            rows["score"].values, scores[name].nlargest(5).values, rtol=1e-6
        )
        assert (
            rows[["x", "y"]].values == keep_run["coords"][rows["tile_index"].values]
        ).all()
    assert (tmp_path / "report.html").read_text().count("<figure") == 5 * len(_PROMPTS)


def test_identity_adapter_leaves_real_keep_scores_unchanged(keep_run, tmp_path):
    """The adapter path with real features: an exact identity adapter must not move a score."""
    d = keep_run["feats"].shape[1]
    torch.save(
        {
            "state_dict": {
                "mean": torch.zeros(d),
                "std": torch.ones(d),
                "net.weight": torch.eye(d),
                "net.bias": torch.zeros(d),
            },
            "kind": "ridge",
            "teacher": "keep",
        },
        tmp_path / "identity.pt",
    )
    plain, _ = _annotate(keep_run, tmp_path / "plain.csv")
    adapted, _ = _annotate(
        keep_run, tmp_path / "adapted.csv", adapter_path=str(tmp_path / "identity.pt")
    )
    np.testing.assert_allclose(adapted.values, plain.values, atol=1e-6)


def test_misuse_fails_clearly(keep_run, tmp_path):
    with pytest.raises(ValueError, match="same order"):
        mussel.cli.annotate.main(
            AnnotateConfig(
                features_pt_path=str(keep_run["tmp"] / "keep.features.pt"),
                class_embedding_pt_path=str(keep_run["tmp"] / "prompts.pt"),
                classes=_PROMPTS[::-1],
                output_csv_path=str(tmp_path / "x.csv"),
            )
        )
