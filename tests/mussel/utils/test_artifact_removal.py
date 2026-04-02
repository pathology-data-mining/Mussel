"""Unit tests for GrandQCArtifactRemover.

These tests mock the underlying segmentation_models_pytorch.Unet and
torch.load so they run without GPU, internet access, or model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


def _make_remover(remove_penmarks_only: bool = False):
    from mussel.utils.artifact_removal import GrandQCArtifactRemover

    return GrandQCArtifactRemover(
        remove_penmarks_only=remove_penmarks_only,
        device="cpu",
        batch_size=4,
    )


def _make_fake_model(predicted_class: int, tile_size: int = 512):
    """Return a mock smp.Unet that always predicts ``predicted_class``."""
    n_classes = 8

    def forward(x):
        b, _, h, w = x.shape
        logits = torch.full((b, n_classes, h, w), -10.0)
        logits[:, predicted_class, :, :] = 10.0
        return logits

    model = MagicMock()
    model.side_effect = forward
    model.__call__ = forward
    model.eval.return_value = model
    model.to.return_value = model
    return model


def _inject_model(remover, predicted_class: int):
    """Bypass _load_model by injecting a fake model directly."""
    from torchvision import transforms

    remover._model = _make_fake_model(predicted_class)
    remover._transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class TestGrandQCArtifactRemoverCore:

    def _run(self, remover, img, mask, mpp=1.0):
        return remover(img, mask, mpp)

    def test_normal_tissue_mask_unchanged(self):
        """When GrandQC predicts normal tissue everywhere, mask is unchanged."""
        remover = _make_remover()
        _inject_model(remover, predicted_class=1)  # 1 = Normal Tissue

        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.ones((512, 512), dtype=np.uint8)

        result = self._run(remover, img, mask)

        np.testing.assert_array_equal(result, mask)

    def test_all_artifact_zeroes_mask(self):
        """When GrandQC predicts all folds (class 2), the full mask is zeroed."""
        remover = _make_remover(remove_penmarks_only=False)
        _inject_model(remover, predicted_class=2)  # 2 = Fold

        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.ones((512, 512), dtype=np.uint8)

        result = self._run(remover, img, mask)

        assert (
            result.sum() == 0
        ), "All tissue should be removed when model predicts folds"

    def test_penmarks_only_keeps_folds(self):
        """remove_penmarks_only=True: fold class (2) is NOT removed."""
        remover = _make_remover(remove_penmarks_only=True)
        _inject_model(remover, predicted_class=2)  # 2 = Fold

        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.ones((512, 512), dtype=np.uint8)

        result = self._run(remover, img, mask)

        np.testing.assert_array_equal(
            result, mask, err_msg="Folds should be kept in penmarks-only mode"
        )

    def test_penmarks_only_removes_penmarks(self):
        """remove_penmarks_only=True: pen marking class (4) IS removed."""
        remover = _make_remover(remove_penmarks_only=True)
        _inject_model(remover, predicted_class=4)  # 4 = Pen Marking

        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.ones((512, 512), dtype=np.uint8)

        result = self._run(remover, img, mask)

        assert result.sum() == 0, "Pen marks should be removed in penmarks-only mode"

    def test_output_shape_matches_input(self):
        """Output mask has the same shape as the input mask."""
        remover = _make_remover()
        _inject_model(remover, predicted_class=1)

        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        mask = np.ones((300, 400), dtype=np.uint8)

        result = self._run(remover, img, mask, mpp=1.0)

        assert result.shape == mask.shape

    def test_output_dtype_preserved(self):
        """Output mask dtype matches input mask dtype."""
        remover = _make_remover()
        _inject_model(remover, predicted_class=1)

        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.ones((512, 512), dtype=np.uint8)

        result = self._run(remover, img, mask)

        assert result.dtype == mask.dtype

    def test_rgba_input_handled(self):
        """RGBA (4-channel) input does not raise an error."""
        remover = _make_remover()
        _inject_model(remover, predicted_class=1)

        img = np.random.randint(0, 255, (512, 512, 4), dtype=np.uint8)
        mask = np.ones((512, 512), dtype=np.uint8)

        result = self._run(remover, img, mask)

        assert result.shape == mask.shape

    def test_non_square_non_tile_multiple_size(self):
        """Images that are not multiples of 512 are handled via padding."""
        remover = _make_remover()
        _inject_model(remover, predicted_class=1)

        img = np.random.randint(0, 255, (700, 900, 3), dtype=np.uint8)
        mask = np.ones((700, 900), dtype=np.uint8)

        result = self._run(remover, img, mask)

        assert result.shape == (700, 900)

    def test_mpp_rescaling_preserves_output_shape(self):
        """At non-target MPP the image is rescaled but output shape is unchanged."""
        remover = _make_remover()
        _inject_model(remover, predicted_class=1)

        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.ones((512, 512), dtype=np.uint8)

        for mpp in (0.25, 0.5, 1.0, 2.0, 4.0):
            result = self._run(remover, img, mask, mpp=mpp)
            assert result.shape == mask.shape, f"Shape mismatch at mpp={mpp}"


class TestGrandQCArtifactRemoverImportError:

    def test_missing_smp_raises_import_error(self):
        """Missing segmentation_models_pytorch raises a clear ImportError."""
        from mussel.utils.artifact_removal import GrandQCArtifactRemover

        remover = GrandQCArtifactRemover(device="cpu")

        with patch.dict("sys.modules", {"segmentation_models_pytorch": None}):
            with pytest.raises(ImportError, match="segmentation-models-pytorch"):
                remover._load_model()
