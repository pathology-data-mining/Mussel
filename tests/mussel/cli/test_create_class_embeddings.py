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
        def encode_text(self, text):
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
