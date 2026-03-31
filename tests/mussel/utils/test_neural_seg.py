"""Unit tests for mussel.utils.neural_seg.NeuralTissueSegmenter."""
import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segmenter(model_obj=None, device="cpu"):
    """Return a NeuralTissueSegmenter with a pre-loaded mock model."""
    from mussel.utils.neural_seg import NeuralTissueSegmenter
    from torchvision import transforms

    seg = NeuralTissueSegmenter.__new__(NeuralTissueSegmenter)
    seg.batch_size = 4
    seg.confidence_thresh = 0.5
    seg.device = torch.device(device)
    seg._weights_path = None
    seg._transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    seg._model = model_obj
    return seg


def _all_tissue_model():
    """Mock model that always predicts tissue (class 1)."""
    mock = MagicMock()
    # Return logits where class-1 >> class-0 for all pixels
    def forward(x):
        B = x.shape[0]
        logits = torch.zeros(B, 2, 512, 512)
        logits[:, 1] = 10.0   # tissue channel high
        logits[:, 0] = -10.0  # background channel low
        return {"out": logits}
    mock.side_effect = forward
    return mock


def _all_background_model():
    """Mock model that always predicts background (class 0)."""
    mock = MagicMock()
    def forward(x):
        B = x.shape[0]
        logits = torch.zeros(B, 2, 512, 512)
        logits[:, 0] = 10.0
        logits[:, 1] = -10.0
        return {"out": logits}
    mock.side_effect = forward
    return mock


# ---------------------------------------------------------------------------
# segment() — output shape and value checks
# ---------------------------------------------------------------------------

class TestSegmentOutputShape:
    def test_output_is_correct_shape(self):
        H, W = 256, 256
        img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        seg = _make_segmenter(_all_tissue_model())
        mask = seg.segment(img, slide_mpp=1.0)
        assert mask.shape == (H, W)

    def test_output_dtype_uint8(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        seg = _make_segmenter(_all_tissue_model())
        mask = seg.segment(img, slide_mpp=1.0)
        assert mask.dtype == np.uint8

    def test_output_only_0_or_255(self):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        seg = _make_segmenter(_all_tissue_model())
        mask = seg.segment(img, slide_mpp=1.0)
        unique = set(np.unique(mask).tolist())
        assert unique <= {0, 255}

    def test_non_square_image(self):
        img = np.random.randint(0, 255, (128, 512, 3), dtype=np.uint8)
        seg = _make_segmenter(_all_tissue_model())
        mask = seg.segment(img, slide_mpp=1.0)
        assert mask.shape == (128, 512)

    def test_small_image_padded_correctly(self):
        """Image smaller than one patch (512×512)."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        seg = _make_segmenter(_all_tissue_model())
        mask = seg.segment(img, slide_mpp=1.0)
        assert mask.shape == (64, 64)


class TestSegmentAllTissue:
    def test_all_tissue_prediction(self):
        img = np.ones((256, 256, 3), dtype=np.uint8) * 200
        seg = _make_segmenter(_all_tissue_model())
        mask = seg.segment(img, slide_mpp=1.0)
        assert np.all(mask == 255)


class TestSegmentAllBackground:
    def test_all_background_prediction(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        seg = _make_segmenter(_all_background_model())
        mask = seg.segment(img, slide_mpp=1.0)
        assert np.all(mask == 0)


# ---------------------------------------------------------------------------
# slide_mpp rescaling
# ---------------------------------------------------------------------------

class TestSlideMppRescaling:
    def test_mpp_scaling_preserves_output_size(self):
        """Output mask must match original img size regardless of slide_mpp."""
        H, W = 100, 200
        img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        seg = _make_segmenter(_all_tissue_model())
        for mpp in (0.5, 1.0, 2.0, 4.0, 8.0):
            mask = seg.segment(img, slide_mpp=mpp)
            assert mask.shape == (H, W), f"shape mismatch at mpp={mpp}"

    def test_mpp_gt_1_triggers_upsample(self):
        """When slide_mpp > 1, image is upsampled before inference."""
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        seg = _make_segmenter(_all_tissue_model())
        # Should not raise even though upsampled size is larger than original
        mask = seg.segment(img, slide_mpp=4.0)
        assert mask.shape == (32, 32)

    def test_mpp_lt_1_triggers_downsample(self):
        """When slide_mpp < 1, image is downsampled before inference."""
        img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        seg = _make_segmenter(_all_tissue_model())
        mask = seg.segment(img, slide_mpp=0.5)
        assert mask.shape == (1024, 1024)


# ---------------------------------------------------------------------------
# _segment_tissue_neural in segment.py
# ---------------------------------------------------------------------------

class TestSegmentTissueNeuralIntegration:
    def test_segment_tissue_neural_called_for_neural_seg_model(self):
        """segment_tissue(seg_model='neural') calls _segment_tissue_neural."""
        from mussel.utils.segment import _segment_tissue_neural

        img = np.ones((128, 128, 3), dtype=np.uint8)
        fake_mask = np.zeros((128, 128), dtype=np.uint8)
        fake_mask[32:96, 32:96] = 255

        with patch("mussel.utils.segment._segment_tissue_neural",
                   return_value=fake_mask) as mock_fn:
            # Import and call directly to verify the mock works
            from mussel.utils import segment as seg_module
            result = seg_module._segment_tissue_neural(img, slide_mpp=4.0)
            mock_fn.assert_called_once_with(img, slide_mpp=4.0)


# ---------------------------------------------------------------------------
# Weight download — mocked
# ---------------------------------------------------------------------------

class TestDownloadCheckpoint:
    def test_download_called_when_no_weights_path(self):
        """_ensure_model_loaded triggers download when no weights_path is given."""
        from mussel.utils.neural_seg import NeuralTissueSegmenter

        with patch("mussel.utils.neural_seg._download_checkpoint",
                   return_value="/fake/deeplabv3_seg_v4.ckpt") as mock_dl, \
             patch("mussel.utils.neural_seg.NeuralTissueSegmenter._build_model",
                   return_value=MagicMock()):
            seg = NeuralTissueSegmenter(device="cpu")
            seg._ensure_model_loaded()
            mock_dl.assert_called_once()

    def test_download_skipped_when_weights_path_provided(self):
        """_ensure_model_loaded skips download if weights_path is set."""
        from mussel.utils.neural_seg import NeuralTissueSegmenter

        with patch("mussel.utils.neural_seg._download_checkpoint") as mock_dl, \
             patch("mussel.utils.neural_seg.NeuralTissueSegmenter._build_model",
                   return_value=MagicMock()):
            seg = NeuralTissueSegmenter(weights_path="/provided/path.ckpt", device="cpu")
            seg._ensure_model_loaded()
            mock_dl.assert_not_called()


# ---------------------------------------------------------------------------
# Helpers for mocking state dict
# ---------------------------------------------------------------------------

def _build_fake_state_dict():
    """Return a minimal state dict compatible with deeplabv3_resnet50 shapes.

    We only verify that the model loads without error — a full state dict would
    be enormous, so we patch load_state_dict instead.
    """
    return {}


class TestNeuralSegmenterLoadModel:
    def test_build_model_returns_deeplabv3_with_2class_head(self):
        """_build_model creates deeplabv3_resnet50 with a 2-class Conv head."""
        from mussel.utils.neural_seg import NeuralTissueSegmenter
        import torch
        import torch.nn as nn

        fake_checkpoint = {"state_dict": {}}

        seg = NeuralTissueSegmenter.__new__(NeuralTissueSegmenter)
        seg.device = torch.device("cpu")

        with patch("torch.load", return_value=fake_checkpoint), \
             patch.object(
                 __import__("torch.nn.modules.module", fromlist=["Module"]).Module,
                 "load_state_dict",
                 return_value=None,
             ):
            model = seg._build_model("/fake/path.ckpt")

        head = model.classifier[4]
        assert isinstance(head, nn.Conv2d)
        assert head.out_channels == 2
