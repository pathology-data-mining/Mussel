import os
from pathlib import Path

import h5py
from mussel.utils import get_features
from mussel.models import ModelType

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def test_get_features():
    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = "tests/testdata/948176.patch.h5"

    patch_h5 = h5py.File(patch_h5_path, "r")
    coords = patch_h5["coords"][:]
    attrs = patch_h5['coords'].attrs

    features, labels = get_features(coords, slide_path, attrs, model_type=ModelType.RESNET50)
    assert features.shape[0] == coords.shape[0]
    assert labels.shape[0] == coords.shape[0]
    assert labels.shape[1] == coords.shape[1]


