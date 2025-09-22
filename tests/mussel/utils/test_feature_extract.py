import os
from pathlib import Path

import h5py
from mussel.utils import get_features, segment_tissue
from mussel.models import ModelType

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def test_get_features():
    slide_path = "tests/testdata/948176.svs"
#    patch_h5_path = "tests/testdata/948176.patch.h5"
#
#    patch_h5 = h5py.File(patch_h5_path, "r")
#    coords = patch_h5["coords"][:]
#    attrs = patch_h5['coords'].attrs

    _, _, coords, attrs = segment_tissue(
        slide_path=slide_path,
        patch_size=224,
        segment_threshold=15,
        median_blur_ksize=11,
        morphology_ex_kernel=2,
        tissue_area_threshold=2,
        hole_area_threshold=1,
        max_num_holes=2
    )

    features, labels = get_features(coords, slide_path, attrs, model_type=ModelType.RESNET50)
    assert features.shape[0] == len(coords)
    assert labels.shape[0] == len(coords)


