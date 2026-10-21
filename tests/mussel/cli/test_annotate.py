import os
from pathlib import Path

from omegaconf import OmegaConf

import mussel.cli.annotate
from mussel.cli.annotate import AnnotateConfig


def test_annotate(tmp_path, test_data_path):
    annotation_classes = [
        "carcinoma in situ",
        "invasive carcinoma",
        "collagenous stroma",
        "adipose",
        "vessel",
        "necrosis",
        "invasive adenocarcinoma",
        "sarcoma",
    ]
    features_pt_path = os.path.join(test_data_path, "948176.features.pt")
    class_embedding_pt_path = os.path.join(test_data_path, "class_embedding.pt")
    output_csv_path = tmp_path / "test.csv"
    cfg = AnnotateConfig(
        features_pt_path=features_pt_path,
        classes=annotation_classes,
        class_embedding_pt_path=class_embedding_pt_path,
        output_csv_path=output_csv_path,
    )
    mussel.cli.annotate.main(cfg)

    assert os.path.exists(output_csv_path)


def _tiles_h5(path, n, test_data_path):
    """A patch H5 with the test slide's first ``n`` tiles (its features cover fewer tiles)."""
    import h5py

    with (
        h5py.File(os.path.join(test_data_path, "948176.patch.h5")) as src,
        h5py.File(path, "w") as dst,
    ):
        ds = dst.create_dataset("coords", data=src["coords"][:n])
        for k, v in src["coords"].attrs.items():
            ds.attrs[k] = v
    return str(path)


def test_annotate_adapter_and_top_k(tmp_path, test_data_path):
    """Features go through the adapter before scoring; top-k lists each class's best tiles."""
    import h5py
    import pandas as pd
    import torch

    features = torch.load(
        os.path.join(test_data_path, "948176.features.pt"), weights_only=True
    )
    class_emb = torch.load(
        os.path.join(test_data_path, "class_embedding.pt"), weights_only=True
    )
    classes = [f"c{i}" for i in range(len(class_emb))]
    torch.manual_seed(0)
    lin = torch.nn.Linear(features.shape[1], class_emb.shape[1])
    mean, std = features.mean(0), features.std(0) + 1e-6
    torch.save(
        {
            "state_dict": {
                "mean": mean,
                "std": std,
                "net.weight": lin.weight.data,
                "net.bias": lin.bias.data,
            },
            "kind": "ridge",
            "teacher": "keep",
        },
        tmp_path / "adapter.pt",
    )
    patch_path = _tiles_h5(tmp_path / "tiles.h5", len(features), test_data_path)
    report = tmp_path / "report.html"
    cfg = AnnotateConfig(
        features_pt_path=os.path.join(test_data_path, "948176.features.pt"),
        classes=classes,
        output_csv_path=str(tmp_path / "s.csv"),
        class_embedding_pt_path=os.path.join(test_data_path, "class_embedding.pt"),
        adapter_path=str(tmp_path / "adapter.pt"),
        top_k=3,
        interrogate=True,
        slide_path=os.path.join(test_data_path, "948176.svs"),
        patch_path=patch_path,
        interrogation_report_path=str(report),
    )
    mussel.cli.annotate.main(cfg)

    with torch.no_grad():
        adapted = torch.nn.functional.normalize(lin((features - mean) / std), dim=-1)
    expected = torch.nn.functional.cosine_similarity(
        adapted.unsqueeze(1), class_emb.unsqueeze(0), dim=2
    )
    scores = pd.read_csv(tmp_path / "s.csv")
    assert list(scores.columns) == classes
    torch.testing.assert_close(
        torch.tensor(scores.values, dtype=torch.float32), expected, atol=1e-5, rtol=1e-4
    )

    top = pd.read_csv(tmp_path / "s.top3.csv")
    with h5py.File(patch_path) as f:
        coords = f["coords"][:]
    assert len(top) == 3 * len(classes)
    for name, rows in top.groupby("class"):
        best = expected[:, classes.index(name)].topk(3).indices.tolist()
        assert rows.sort_values("rank")["tile_index"].tolist() == best
        assert rows["score"].is_monotonic_decreasing
        assert (rows[["x", "y"]].values == coords[rows["tile_index"].values]).all()
    html = report.read_text()
    assert html.count("<figure") == 3 * len(classes) and html.count("<h2>") == len(
        classes
    )


def _annotate_cfg(tmp_path, test_data_path, classes, emb_path):
    return AnnotateConfig(features_pt_path=os.path.join(test_data_path, "948176.features.pt"),
                          class_embedding_pt_path=str(emb_path), classes=classes,
                          output_csv_path=str(tmp_path / "s.csv"))


def test_annotate_rejects_classes_that_do_not_match_the_embeddings(tmp_path, test_data_path):
    import json

    import pytest
    import torch

    emb = torch.load(os.path.join(test_data_path, "class_embedding.pt"), weights_only=True)
    names = [f"c{i}" for i in range(len(emb))]
    p = tmp_path / "e.pt"
    torch.save(emb, p)
    (tmp_path / "e.pt.classes.json").write_text(json.dumps({"classes": names, "model_type": "CLIP"}))
    with pytest.raises(ValueError, match="same order"):
        mussel.cli.annotate.main(_annotate_cfg(tmp_path, test_data_path, names[::-1], p))
    with pytest.raises(ValueError, match="classes given"):
        mussel.cli.annotate.main(_annotate_cfg(tmp_path, test_data_path, names[:-1], p))
    mussel.cli.annotate.main(_annotate_cfg(tmp_path, test_data_path, names, p))  # matching list works


def test_annotate_rejects_features_of_another_width(tmp_path, test_data_path):
    import pytest
    import torch

    p = tmp_path / "keep.pt"
    torch.save(torch.nn.functional.normalize(torch.randn(3, 768), dim=-1), p)  # e.g. KEEP prompts
    with pytest.raises(ValueError, match="adapter_path"):
        mussel.cli.annotate.main(_annotate_cfg(tmp_path, test_data_path, ["a", "b", "c"], p))
