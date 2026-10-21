import torch

import mussel.cli.create_class_embeddings
from mussel.cli.create_class_embeddings import ClassEmbeddingConfig


def test_create_class_embeddings(tmp_path, monkeypatch):
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
    output_pt_path = tmp_path / "test.pt"

    class FakeModel:
        training = True  # open_clip returns models in training mode (dropout active)

        def train(self, mode=True):
            self.training = mode
            return self

        def encode_text(self, text):
            assert not self.training, "class embeddings must be computed in eval mode"
            return torch.arange(4, dtype=torch.float32).unsqueeze(0) + text.float()

    def fake_create_model_and_transforms(model_path):
        return FakeModel(), None, None

    def fake_get_tokenizer(model_path):
        return lambda class_text: torch.tensor([[len(class_text)]], dtype=torch.float32)

    monkeypatch.setattr(
        mussel.cli.create_class_embeddings.open_clip,
        "create_model_and_transforms",
        fake_create_model_and_transforms,
    )
    monkeypatch.setattr(
        mussel.cli.create_class_embeddings.open_clip,
        "get_tokenizer",
        fake_get_tokenizer,
    )

    cfg = ClassEmbeddingConfig(
        classes=annotation_classes, output_pt_path=output_pt_path
    )
    mussel.cli.create_class_embeddings.main(cfg)
    assert output_pt_path.exists()
    class_emb = torch.load(output_pt_path, weights_only=True)
    assert class_emb.shape == (len(annotation_classes), 4)


def test_create_class_embeddings_keep(tmp_path, monkeypatch):
    """model_type=KEEP encodes the classes with KEEP's text tower."""
    import mussel.models.keep
    from mussel.models.model_factory import ModelType

    seen = {}

    class FakeKeepText:
        def __init__(self, model_path):
            seen["model_path"] = model_path

        def encode(self, texts):
            seen["texts"] = texts
            return torch.nn.functional.normalize(
                torch.arange(len(texts) * 4, dtype=torch.float32).view(-1, 4) + 1,
                dim=-1,
            )

    monkeypatch.setattr(mussel.models.keep, "KeepTextEncoder", FakeKeepText)
    out = tmp_path / "keep.pt"
    mussel.cli.create_class_embeddings.main(
        ClassEmbeddingConfig(
            classes=["tumor", "stroma", "fat"],
            output_pt_path=str(out),
            model_type=ModelType.KEEP,
            model_path="/models/keep",
        )
    )
    emb = torch.load(out, weights_only=True)
    assert seen == {"model_path": "/models/keep", "texts": ["tumor", "stroma", "fat"]}
    assert emb.shape == (3, 4)


def test_create_class_embeddings_writes_the_class_list_beside_the_embeddings(tmp_path, monkeypatch):
    import json

    import mussel.models.keep
    from mussel.models.model_factory import ModelType

    class FakeKeepText:
        def __init__(self, model_path):
            pass

        def encode(self, texts):
            return torch.ones(len(texts), 4)

    monkeypatch.setattr(mussel.models.keep, "KeepTextEncoder", FakeKeepText)
    out = tmp_path / "keep.pt"
    mussel.cli.create_class_embeddings.main(ClassEmbeddingConfig(
        classes=["tumor", "stroma"], output_pt_path=str(out), model_type=ModelType.KEEP))
    assert json.loads((tmp_path / "keep.pt.classes.json").read_text()) == {
        "classes": ["tumor", "stroma"], "model_type": "KEEP"}


def test_create_class_embeddings_rejects_models_without_a_supported_text_tower(tmp_path):
    import pytest

    from mussel.models.model_factory import ModelType

    with pytest.raises(ValueError, match="CLIP or KEEP"):
        mussel.cli.create_class_embeddings.main(ClassEmbeddingConfig(
            classes=["tumor"], output_pt_path=str(tmp_path / "x.pt"), model_type=ModelType.MUSK))
