"""Test that WholeSlideImageTileCoordDataset produces equivalent outputs to WholeSlideImageH5Dataset."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from mussel.datasets.h5 import WholeSlideImageH5Dataset
from mussel.datasets.tile_coords import WholeSlideImageTileCoordDataset

TESTDATA_DIR = Path(__file__).parent.parent.parent / "testdata"
H5_PATH = TESTDATA_DIR / "948176.patch.h5"
SLIDE_PATH = TESTDATA_DIR / "948176.svs"


@pytest.fixture
def h5_dataset():
    """Create H5 dataset from test data."""
    if not H5_PATH.exists() or not SLIDE_PATH.exists():
        pytest.skip("Test data not available")
    return WholeSlideImageH5Dataset(
        h5_path=str(H5_PATH),
        slide_path=str(SLIDE_PATH),
        use_imagenet_rgb_dist=True,
        init_wsi_in_worker=False,
    )


@pytest.fixture
def tile_coord_dataset():
    """Create TileCoord dataset from the same H5 file's coordinates."""
    if not H5_PATH.exists() or not SLIDE_PATH.exists():
        pytest.skip("Test data not available")

    # Load coords and attrs from h5 file
    with h5py.File(H5_PATH, "r") as f:
        coords = f["coords"][:]
        attrs = {
            "patch_size": f["coords"].attrs["patch_size"],
            "patch_level": f["coords"].attrs["patch_level"],
            "patch_size_to_resize_to_for_desired_mpp": f["coords"].attrs[
                "patch_size_to_resize_to_for_desired_mpp"
            ],
        }

    return WholeSlideImageTileCoordDataset(
        coords=coords,
        attrs=attrs,
        slide_path=str(SLIDE_PATH),
        use_imagenet_rgb_dist=True,
        init_wsi_in_worker=False,
    )


def test_datasets_have_same_length(h5_dataset, tile_coord_dataset):
    """Both datasets should have the same number of tiles."""
    assert len(h5_dataset) == len(tile_coord_dataset)


def test_datasets_have_same_attributes(h5_dataset, tile_coord_dataset):
    """Both datasets should have identical patch attributes."""
    assert h5_dataset.patch_size == tile_coord_dataset.patch_size
    assert h5_dataset.patch_level == tile_coord_dataset.patch_level
    assert h5_dataset.scaled_patch_size == tile_coord_dataset.scaled_patch_size


def test_datasets_have_same_transforms(h5_dataset, tile_coord_dataset):
    """Both datasets should use the same transforms."""
    assert h5_dataset.use_imagenet_rgb_dist == tile_coord_dataset.use_imagenet_rgb_dist
    assert str(h5_dataset.roi_transforms) == str(tile_coord_dataset.roi_transforms)


def test_datasets_return_identical_tiles(h5_dataset, tile_coord_dataset):
    """Both datasets should return identical image tensors and coordinates for the same indices."""
    # Test a sample of indices to keep test fast
    num_samples = min(5, len(h5_dataset))
    indices = [0, len(h5_dataset) // 2, len(h5_dataset) - 1][:num_samples]

    for idx in indices:
        h5_img, h5_coord = h5_dataset[idx]
        tc_img, tc_coord = tile_coord_dataset[idx]

        # Skip if either returned None (corrupted tile)
        if h5_img is None or tc_img is None:
            continue

        # Coordinates should be identical
        np.testing.assert_array_equal(
            h5_coord, tc_coord, err_msg=f"Coordinates differ at index {idx}"
        )

        # Image tensors should be identical
        torch.testing.assert_close(
            h5_img, tc_img, msg=f"Image tensors differ at index {idx}"
        )


def test_datasets_with_limit_to_indices(h5_dataset, tile_coord_dataset):
    """Both datasets should handle limit_to_indices identically."""
    if len(h5_dataset) < 3:
        pytest.skip("Not enough tiles to test limit_to_indices")

    limit_indices = [0, 2]

    # Load coords and attrs from h5 file for tile_coord dataset
    with h5py.File(H5_PATH, "r") as f:
        coords = f["coords"][:]
        attrs = {
            "patch_size": f["coords"].attrs["patch_size"],
            "patch_level": f["coords"].attrs["patch_level"],
            "patch_size_to_resize_to_for_desired_mpp": f["coords"].attrs[
                "patch_size_to_resize_to_for_desired_mpp"
            ],
        }

    h5_limited = WholeSlideImageH5Dataset(
        h5_path=str(H5_PATH),
        slide_path=str(SLIDE_PATH),
        use_imagenet_rgb_dist=True,
        limit_to_indices=limit_indices,
        init_wsi_in_worker=False,
    )

    tc_limited = WholeSlideImageTileCoordDataset(
        coords=coords,
        attrs=attrs,
        slide_path=str(SLIDE_PATH),
        use_imagenet_rgb_dist=True,
        limit_to_indices=limit_indices,
        init_wsi_in_worker=False,
    )

    assert len(h5_limited) == len(tc_limited) == len(limit_indices)

    for idx in range(len(limit_indices)):
        h5_img, h5_coord = h5_limited[idx]
        tc_img, tc_coord = tc_limited[idx]

        if h5_img is None or tc_img is None:
            continue

        np.testing.assert_array_equal(h5_coord, tc_coord)
        torch.testing.assert_close(h5_img, tc_img)


def test_datasets_without_imagenet_normalization():
    """Both datasets should behave identically without ImageNet normalization."""
    if not H5_PATH.exists() or not SLIDE_PATH.exists():
        pytest.skip("Test data not available")

    with h5py.File(H5_PATH, "r") as f:
        coords = f["coords"][:]
        attrs = {
            "patch_size": f["coords"].attrs["patch_size"],
            "patch_level": f["coords"].attrs["patch_level"],
            "patch_size_to_resize_to_for_desired_mpp": f["coords"].attrs[
                "patch_size_to_resize_to_for_desired_mpp"
            ],
        }

    h5_ds = WholeSlideImageH5Dataset(
        h5_path=str(H5_PATH),
        slide_path=str(SLIDE_PATH),
        use_imagenet_rgb_dist=False,
        init_wsi_in_worker=False,
    )

    tc_ds = WholeSlideImageTileCoordDataset(
        coords=coords,
        attrs=attrs,
        slide_path=str(SLIDE_PATH),
        use_imagenet_rgb_dist=False,
        init_wsi_in_worker=False,
    )

    h5_img, h5_coord = h5_ds[0]
    tc_img, tc_coord = tc_ds[0]

    if h5_img is not None and tc_img is not None:
        np.testing.assert_array_equal(h5_coord, tc_coord)
        torch.testing.assert_close(h5_img, tc_img)
