import os
from omegaconf import OmegaConf

import mussel.cli.annotate
from mussel.cli.annotate import AnnotateConfig

# Import fixtures from common conftest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import test_data_path

def test_annotate(tmp_path, test_data_path):
    annotation_classes = [
        "carcinoma in situ",
        "invasive carcinoma",
        "collagenous stroma",
        "adipose",
        "vessel",
        "necrosis",
        "invasive adenocarcinoma",
        "sarcoma"]
    features_pt_path = os.path.join(test_data_path, "948176.features.pt")
    class_embedding_pt_path = os.path.join(test_data_path, "class_embedding.pt")
    output_csv_path = tmp_path / "test.csv"
    cfg = AnnotateConfig(
        features_pt_path=features_pt_path,
        classes=annotation_classes,
        class_embedding_pt_path=class_embedding_pt_path,
        output_csv_path=output_csv_path
    )
    mussel.cli.annotate.main(cfg)

    assert os.path.exists(output_csv_path)

