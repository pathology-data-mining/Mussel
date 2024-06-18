import os
from omegaconf import OmegaConf

import mussel.cli.create_class_embeddings
from mussel.cli.create_class_embeddings import ClassEmbeddingConfig

def test_create_class_embeddings(tmp_path):
    annotation_classes = ["carcinoma in situ",
                        "invasive carcinoma",
                        "collagenous stroma",
                        "adipose",
                        "vessel",
                        "necrosis",
                        "invasive adenocarcinoma",
                        "sarcoma"]
    output_pt_path = tmp_path / "test.pt"
    cfg = ClassEmbeddingConfig(classes=annotation_classes, output_pt_path=output_pt_path)
    mussel.cli.create_class_embeddings.main(cfg)
    assert os.path.exists(output_pt_path)
