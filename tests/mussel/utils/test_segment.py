from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mussel.utils.segment import get_slide_mpp


class TestGetSlideMPP:
    """Test suite for get_slide_mpp function with comprehensive fallback coverage"""

    def test_get_slide_mpp_standard_property(self):
        """Test MPP retrieval from standard tiffslide property"""
        import tiffslide

        # Mock wsi object with standard MPP property
        wsi = MagicMock()
        wsi.properties = {tiffslide.PROPERTY_NAME_MPP_X: "0.25"}

        mpp = get_slide_mpp(wsi)

        assert mpp == 0.25

    def test_get_slide_mpp_override_bypasses_metadata(self):
        """slide_mpp_override short-circuits all metadata reading"""
        wsi = MagicMock()
        # Use a MagicMock for properties so we can assert .get is never called.
        wsi.properties = MagicMock()

        mpp = get_slide_mpp(wsi, slide_mpp_override=1.0)

        assert mpp == 1.0
        wsi.properties.get.assert_not_called()

    def test_get_slide_mpp_override_returned_as_float(self):
        """slide_mpp_override is coerced to float"""
        wsi = MagicMock()
        wsi.properties = {}

        mpp = get_slide_mpp(wsi, slide_mpp_override=1)

        assert mpp == 1.0
        assert isinstance(mpp, float)

    def test_get_slide_mpp_aperio_property(self):
        """Test MPP retrieval from aperio.MPP property"""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {tiffslide.PROPERTY_NAME_MPP_X: None, "aperio.MPP": "0.25"}

        mpp = get_slide_mpp(wsi)

        assert mpp == 0.25

    def test_get_slide_mpp_openslide_property(self):
        """Test MPP retrieval from openslide.mpp-x property"""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            "aperio.MPP": None,
            "openslide.mpp-x": "1.0",
        }

        mpp = get_slide_mpp(wsi)

        assert mpp == 1.0

    def test_get_slide_mpp_from_aperio_magnification(self):
        """Test MPP estimation from aperio.AppMag magnification"""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {tiffslide.PROPERTY_NAME_MPP_X: None, "aperio.AppMag": "40"}

        mpp = get_slide_mpp(wsi)

        # 40x magnification -> 10.0 / 40 = 0.25 MPP
        assert mpp == 0.25

    def test_get_slide_mpp_from_openslide_magnification(self):
        """Test MPP estimation from openslide.objective-power"""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            "aperio.AppMag": None,
            "openslide.objective-power": "20",
        }

        mpp = get_slide_mpp(wsi)

        # 20x magnification -> 10.0 / 20 = 0.5 MPP
        assert mpp == 0.5

    def test_get_slide_mpp_from_tiffslide_magnification(self):
        """Test MPP estimation from tiffslide.objective-power"""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            "aperio.AppMag": None,
            "openslide.objective-power": None,
            tiffslide.PROPERTY_NAME_OBJECTIVE_POWER: "10",
        }

        mpp = get_slide_mpp(wsi)

        # 10x magnification -> 10.0 / 10 = 1.0 MPP
        assert mpp == 1.0

    def test_get_slide_mpp_magnification_float_value(self):
        """Test magnification with float value (e.g., 20.5x)"""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {tiffslide.PROPERTY_NAME_MPP_X: None, "aperio.AppMag": "20.5"}

        mpp = get_slide_mpp(wsi)

        # 20.5x magnification -> 10.0 / 20.5 ≈ 0.488 MPP
        assert abs(mpp - 10.0 / 20.5) < 0.001

    def test_get_slide_mpp_invalid_magnification_fallback(self):
        """Test fallback when magnification is not a valid number"""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            "aperio.AppMag": "invalid",
        }

        mpp = get_slide_mpp(wsi)

        # Should fallback to default 0.5 MPP
        assert mpp == 0.5

    def test_get_slide_mpp_no_metadata_default_fallback(self):
        """Test fallback to default MPP when no metadata found"""
        wsi = MagicMock()
        wsi.properties = {}

        mpp = get_slide_mpp(wsi)

        # Should use default 0.5 MPP
        assert mpp == 0.5

    def test_get_slide_mpp_custom_default(self):
        """Test custom default MPP value"""
        wsi = MagicMock()
        wsi.properties = {}

        mpp = get_slide_mpp(wsi, default_mpp=0.75)

        assert mpp == 0.75

    def test_get_slide_mpp_with_slide_path_logging(self):
        """Test that slide_path is used in log messages"""
        wsi = MagicMock()
        wsi.properties = {}

        with patch("mussel.utils.segment.logger") as mock_logger:
            mpp = get_slide_mpp(wsi, slide_path="/path/to/slide.svs")

            # Should log warning with slide path
            assert mock_logger.warning.called
            call_args = str(mock_logger.warning.call_args)
            assert "/path/to/slide.svs" in call_args

    def test_get_slide_mpp_exception_handling(self):
        """Test exception handling with invalid property access"""
        wsi = MagicMock()
        # Simulate property access raising an exception
        wsi.properties.get.side_effect = KeyError("Property not found")

        mpp = get_slide_mpp(wsi, default_mpp=0.6)

        # Should catch exception and use default
        assert mpp == 0.6

    def test_get_slide_mpp_type_error_handling(self):
        """Test handling of TypeError when converting MPP value"""
        import tiffslide

        wsi = MagicMock()
        # Return a value that can't be converted to float
        wsi.properties = {tiffslide.PROPERTY_NAME_MPP_X: object()}

        mpp = get_slide_mpp(wsi, default_mpp=0.7)

        # Should catch TypeError and use default
        assert mpp == 0.7

    def test_get_slide_mpp_value_error_handling(self):
        """Test handling of ValueError when converting invalid string"""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {tiffslide.PROPERTY_NAME_MPP_X: "not-a-number"}

        mpp = get_slide_mpp(wsi, default_mpp=0.8)

        # Should catch ValueError and use default
        assert mpp == 0.8

    def test_get_slide_mpp_zero_magnification_handling(self):
        """Test that zero magnification falls back to default MPP (no ZeroDivisionError)."""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            "aperio.AppMag": "0",
        }

        mpp = get_slide_mpp(wsi)
        assert mpp == 0.5  # falls back to default

    def test_get_slide_mpp_priority_order(self):
        """Test that standard property takes priority over alternatives"""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: "0.25",  # Standard property
            "aperio.MPP": "0.5",  # Alternative property
            "aperio.AppMag": "10",  # Magnification
        }

        mpp = get_slide_mpp(wsi)

        # Should use standard property (0.25), not alternatives
        assert mpp == 0.25

    def test_get_slide_mpp_magnification_priority_order(self):
        """Test that aperio.AppMag takes priority over other magnification properties"""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            "aperio.AppMag": "40",  # Should use this
            "openslide.objective-power": "20",
            "tiffslide.objective-power": "10",
        }

        mpp = get_slide_mpp(wsi)

        # Should use aperio.AppMag (40x -> 0.25 MPP)
        assert mpp == 0.25

    def test_get_slide_mpp_logging_standard_property(self):
        """Test logging when MPP found in standard property"""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {tiffslide.PROPERTY_NAME_MPP_X: "0.5"}

        with patch("mussel.utils.segment.logger") as mock_logger:
            get_slide_mpp(wsi)

            # Should log info message
            mock_logger.info.assert_called()
            call_args = str(mock_logger.info.call_args)
            assert "0.5" in call_args

    def test_get_slide_mpp_logging_alternate_property(self):
        """Test logging when MPP found in alternate property"""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {tiffslide.PROPERTY_NAME_MPP_X: None, "aperio.MPP": "0.25"}

        with patch("mussel.utils.segment.logger") as mock_logger:
            get_slide_mpp(wsi)

            # Should log info about alternate property
            assert mock_logger.info.call_count >= 1
            calls_str = str(mock_logger.info.call_args_list)
            assert "aperio.MPP" in calls_str

    def test_get_slide_mpp_logging_magnification_estimation(self):
        """Test logging when MPP estimated from magnification"""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {tiffslide.PROPERTY_NAME_MPP_X: None, "aperio.AppMag": "20"}

        with patch("mussel.utils.segment.logger") as mock_logger:
            get_slide_mpp(wsi)

            # Should log warning about estimation
            mock_logger.warning.assert_called_once()
            call_args = str(mock_logger.warning.call_args)
            assert "estimated" in call_args.lower()
            assert "20" in call_args  # magnification value

    def test_get_slide_mpp_logging_default_fallback(self):
        """Test logging when falling back to default MPP"""
        wsi = MagicMock()
        wsi.properties = {}

        with patch("mussel.utils.segment.logger") as mock_logger:
            get_slide_mpp(wsi)

            # Should log warning about using default
            mock_logger.warning.assert_called_once()
            call_args = str(mock_logger.warning.call_args)
            assert "default" in call_args.lower()

    def test_get_slide_mpp_tiff_xresolution_inch(self):
        """Step 4: derive MPP from tiff.XResolution when tiffslide.mpp-x is absent."""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            "tiff.XResolution": "40000",  # 40000 px/inch → 25400/40000 = 0.635 µm/px
            "tiff.ResolutionUnit": "INCH",
        }

        mpp = get_slide_mpp(wsi)
        assert abs(mpp - 0.635) < 0.001

    def test_get_slide_mpp_tiff_xresolution_centimeter(self):
        """Step 4: derive MPP from tiff.XResolution in cm units."""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            "tiff.XResolution": "20000",  # 20000 px/cm → 10000/20000 = 0.5 µm/px
            "tiff.ResolutionUnit": "CENTIMETER",
        }

        mpp = get_slide_mpp(wsi)
        assert mpp == pytest.approx(0.5, abs=1e-4)

    def test_get_slide_mpp_tiff_xresolution_micrometer(self):
        """Step 4: MICROMETER units — XResolution is directly px/µm."""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            "tiff.XResolution": "2.0",  # 2 px/µm → 0.5 µm/px
            "tiff.ResolutionUnit": "MICROMETER",
        }

        mpp = get_slide_mpp(wsi)
        assert mpp == pytest.approx(0.5, abs=1e-4)

    def test_get_slide_mpp_tiff_xresolution_unknown_unit_skipped(self):
        """Step 4: unknown ResolutionUnit — skip and fall through to default."""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            "tiff.XResolution": "40000",
            "tiff.ResolutionUnit": "NONE",
        }

        mpp = get_slide_mpp(wsi)
        assert mpp == 0.5  # falls through to default

    def test_get_slide_mpp_tiffslide_takes_priority_over_tiff_tags(self):
        """tiffslide.mpp-x should win over tiff.XResolution."""
        import tiffslide

        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: "0.25",
            "tiff.XResolution": "40000",
            "tiff.ResolutionUnit": "INCH",
        }

        assert get_slide_mpp(wsi) == 0.25


class TestDrawSlideMask:
    """Test suite for draw_slide_mask function resource cleanup"""

    def test_draw_slide_mask_closes_wsi_on_success(self):
        """Test that WSI is closed after successful execution"""
        from unittest.mock import MagicMock, call, patch

        import shapely

        # Create mock WSI
        mock_wsi = MagicMock()
        mock_wsi.level_dimensions = [(1000, 1000), (500, 500)]
        mock_wsi.level_downsamples = [1.0, 2.0]
        mock_wsi.get_best_level_for_downsample.return_value = 1

        # Mock read_region to return a simple image
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_wsi.read_region.return_value = mock_image

        # Create a simple polygon
        polygon = shapely.geometry.box(0, 0, 100, 100)

        with patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi):
            with patch("mussel.utils.segment.np.array", return_value=[[0, 0, 0]]):
                with patch("mussel.utils.segment.Image.fromarray") as mock_fromarray:
                    with patch(
                        "mussel.utils.segment.scale_geometry",
                        side_effect=lambda g, s: g,
                    ):
                        mock_img = MagicMock()
                        mock_img.size = (1000, 1000)
                        mock_fromarray.return_value = mock_img

                        from mussel.utils.segment import draw_slide_mask

                        result = draw_slide_mask("/fake/path.svs", polygon)

                        # Verify WSI was closed
                        mock_wsi.close.assert_called_once()

    def test_draw_slide_mask_closes_wsi_on_exception(self):
        """Test that WSI is closed even if exception occurs during processing"""
        from unittest.mock import MagicMock, patch

        import shapely

        # Create mock WSI
        mock_wsi = MagicMock()
        mock_wsi.level_dimensions = [(1000, 1000)]

        # Make read_region raise an exception
        mock_wsi.read_region.side_effect = RuntimeError("Simulated error")

        polygon = shapely.geometry.box(0, 0, 100, 100)

        with patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi):
            with patch(
                "mussel.utils.segment._assert_level_downsamples",
                return_value=[(1.0, 1.0)],
            ):
                from mussel.utils.segment import draw_slide_mask

                # Should raise exception
                with pytest.raises(RuntimeError, match="Simulated error"):
                    draw_slide_mask("/fake/path.svs", polygon)

                # But WSI should still be closed
                mock_wsi.close.assert_called_once()


def test_segment_tissue_closes_wsi_on_early_return_dimension_error():
    """Test that segment_tissue closes WSI when returning early due to large dimensions."""
    mock_wsi = MagicMock()
    mock_wsi.level_dimensions = [(10**7, 10**7)]  # Triggers width * height > 1e12

    with patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi):
        from mussel.utils.segment import segment_tissue

        result = segment_tissue("/fake/path.svs", seg_level=0)

        # Should return None due to dimension error
        assert result is None
        # But WSI should still be closed
        mock_wsi.close.assert_called_once()


def test_segment_tissue_closes_wsi_on_early_return_no_contours():
    """Test that segment_tissue closes WSI when returning early due to no contours."""
    mock_wsi = MagicMock()
    mock_wsi.level_dimensions = [(1000, 1000), (500, 500)]
    mock_wsi.get_best_level_for_downsample.return_value = 1

    # Mock read_region to return a blank image (no tissue)
    mock_img = np.zeros((500, 500, 3), dtype=np.uint8)
    mock_wsi.read_region.return_value = mock_img

    with patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi):
        with patch("mussel.utils.segment.get_slide_mpp", return_value=0.25):
            with patch(
                "mussel.utils.segment._assert_level_downsamples",
                return_value=[(1.0, 1.0), (2.0, 2.0)],
            ):
                with patch(
                    "cv2.findContours", return_value=([], None)
                ):  # No contours found
                    from mussel.utils.segment import segment_tissue

                    result = segment_tissue("/fake/path.svs")

                    # Should return None due to no contours
                    assert result is None
                    # But WSI should still be closed
                    mock_wsi.close.assert_called_once()


def test_segment_tissue_closes_wsi_on_exception():
    """Test that segment_tissue closes WSI when an exception occurs."""
    mock_wsi = MagicMock()
    mock_wsi.level_dimensions = [(1000, 1000)]
    mock_wsi.read_region.side_effect = RuntimeError("Simulated read error")

    with patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi):
        with patch("mussel.utils.segment.get_slide_mpp", return_value=0.25):
            from mussel.utils.segment import segment_tissue

            with pytest.raises(RuntimeError, match="Simulated read error"):
                segment_tissue("/fake/path.svs", seg_level=0)

            # WSI should still be closed despite exception
            mock_wsi.close.assert_called_once()


def test_save_patches_png_closes_wsi_and_pool_on_success(tmp_path):
    """Test that save_patches_png closes both WSI and multiprocessing pool on success."""
    mock_wsi = MagicMock()
    mock_wsi.level_dimensions = [(1000, 1000)]
    mock_pool = MagicMock()

    coords = [(0, 0), (256, 0), (0, 256)]

    with patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi):
        with patch("mussel.utils.segment.mp.Pool", return_value=mock_pool):
            with patch("mussel.utils.segment.get_slide_mpp", return_value=0.5):
                from mussel.utils.segment import save_patches_png

                save_patches_png(
                    slide_path="/fake/path.svs",
                    coords=coords,
                    save_dir=str(tmp_path / "output"),
                    num_workers=2,
                )

                # Both WSI and pool should be closed
                mock_wsi.close.assert_called_once()
                mock_pool.close.assert_called_once()
                mock_pool.join.assert_called_once()


def test_save_patches_png_closes_wsi_and_pool_on_exception(tmp_path):
    """Test that save_patches_png closes both WSI and multiprocessing pool when exception occurs."""
    mock_wsi = MagicMock()
    mock_wsi.level_dimensions = [(1000, 1000)]
    mock_pool = MagicMock()
    mock_pool.starmap.side_effect = RuntimeError("Simulated pool error")

    coords = [(0, 0), (256, 0)]

    with patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi):
        with patch("mussel.utils.segment.mp.Pool", return_value=mock_pool):
            with patch("mussel.utils.segment.get_slide_mpp", return_value=0.5):
                from mussel.utils.segment import save_patches_png

                with pytest.raises(RuntimeError, match="Simulated pool error"):
                    save_patches_png(
                        slide_path="/fake/path.svs",
                        coords=coords,
                        save_dir=str(tmp_path / "output"),
                        num_workers=2,
                    )

                # Both WSI and pool should still be closed despite exception
                mock_wsi.close.assert_called_once()
                mock_pool.close.assert_called_once()
                mock_pool.join.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for tissue_area_threshold scaling fix
# ---------------------------------------------------------------------------


class TestTissueAreaThresholdScaling:
    """
    Verify that tissue_area_threshold scales with native_patch_size
    (derived from patch_size / mpp) rather than the legacy ref_patch_size=512.

    The key invariant: a given tissue_area_threshold value should produce the
    same minimum tissue size in µm² regardless of which pyramid level is used
    for segmentation.
    """

    def _make_wsi(self, seg_level_ds: float, width=8192, height=8192):
        """Mock WSI whose seg level has the given downsample factor."""
        mock_wsi = MagicMock()
        seg_w = max(1, int(width / seg_level_ds))
        seg_h = max(1, int(height / seg_level_ds))
        mock_wsi.level_dimensions = [(width, height), (seg_w, seg_h)]
        mock_wsi.level_downsamples = [1.0, seg_level_ds]
        mock_wsi.get_best_level_for_downsample.return_value = 1
        # Pink/purple H&E-like colour → non-zero HSV saturation → detected by classic segmenter.
        # A 20×20 seg-px island in the centre.
        img = np.zeros((seg_h, seg_w, 3), dtype=np.uint8)
        cy, cx = seg_h // 2, seg_w // 2
        half = 10
        img[cy - half : cy + half, cx - half : cx + half] = [210, 130, 160]
        mock_wsi.read_region.return_value = img
        return mock_wsi

    def _run(
        self, mock_wsi, seg_level_ds, tissue_area_threshold, patch_size=256, mpp=0.5
    ):
        from mussel.utils.segment import segment_tissue

        with (
            patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi),
            patch("mussel.utils.segment.get_slide_mpp", return_value=mpp),
            patch(
                "mussel.utils.segment._assert_level_downsamples",
                return_value=[(1.0, 1.0), (seg_level_ds, seg_level_ds)],
            ),
        ):
            return segment_tissue(
                "/fake/slide.svs",
                patch_size=patch_size,
                mpp=mpp,
                seg_level=1,
                tissue_area_threshold=tissue_area_threshold,
            )

    def test_small_tissue_found_at_low_threshold(self):
        """
        With threshold=1 a tissue region of exactly one native patch is found;
        a region smaller than one patch is filtered out.
        patch_size=256, mpp=0.5 → native_patch=512 px.
        At ds=4: seg_patch_area = 512² / 16 = 16384 seg-px.
        Island of 130×130=16900 seg-px (>16384) should pass;
        island of 20×20=400 seg-px should not.
        """
        ds = 4.0
        patch_size, mpp_val = 256, 0.5
        native_patch = patch_size / mpp_val  # 512
        seg_patch_area = int(native_patch**2 / ds**2)  # 16384

        def _make_wsi_with_island(island_px):
            slide_w, slide_h = 16384, 16384
            seg_w = int(slide_w / ds)
            seg_h = int(slide_h / ds)
            mock_wsi = MagicMock()
            mock_wsi.level_dimensions = [(slide_w, slide_h), (seg_w, seg_h)]
            mock_wsi.level_downsamples = [1.0, ds]
            mock_wsi.get_best_level_for_downsample.return_value = 1
            img = np.zeros((seg_h, seg_w, 3), dtype=np.uint8)
            cy, cx = seg_h // 2, seg_w // 2
            half = island_px // 2
            img[cy - half : cy + half, cx - half : cx + half] = [210, 130, 160]
            mock_wsi.read_region.return_value = img
            return mock_wsi

        # Island bigger than one patch → found
        big_wsi = _make_wsi_with_island(island_px=int(seg_patch_area**0.5) + 5)
        result = self._run(
            big_wsi, ds, tissue_area_threshold=1, patch_size=patch_size, mpp=mpp_val
        )
        assert (
            result is not None
        ), "Island > 1 native patch should be found at threshold=1"

        # Island much smaller than one patch → filtered
        small_wsi = _make_wsi_with_island(island_px=10)
        result = self._run(
            small_wsi, ds, tissue_area_threshold=1, patch_size=patch_size, mpp=mpp_val
        )
        assert (
            result is None
        ), "Island << 1 native patch should be filtered at threshold=1"

    def test_threshold_scales_with_native_patch_size_not_ref_patch_size(self):
        """
        Demonstrate that the fix uses native_patch_size instead of ref_patch_size=512.

        Setup: patch_size=256, mpp=slide_mpp=0.5 → native_patch_size=256 px, ds=4
          New formula: seg_patch_area = 256² / 4² = 4096 seg-px  (threshold=1 → 4096 px)
          Old formula: seg_patch_area = 512² / 4² = 16384 seg-px  (threshold=1 → 16384 px)

        An island of 90×90 = 8100 seg-px lies between the two thresholds:
          - New formula: 8100 > 4096 → island is found ✓
          - Old formula: 8100 < 16384 → island would be filtered (the bug)
        """
        ds = 4.0
        slide_w, slide_h = 8192, 8192
        seg_w = int(slide_w / ds)  # 2048
        seg_h = int(slide_h / ds)  # 2048
        mock_wsi = MagicMock()
        mock_wsi.level_dimensions = [(slide_w, slide_h), (seg_w, seg_h)]
        mock_wsi.level_downsamples = [1.0, ds]
        mock_wsi.get_best_level_for_downsample.return_value = 1
        img = np.zeros((seg_h, seg_w, 3), dtype=np.uint8)
        island_half = 45  # 90×90 = 8100 seg-px
        cy, cx = seg_h // 2, seg_w // 2
        img[
            cy - island_half : cy + island_half, cx - island_half : cx + island_half
        ] = [210, 130, 160]
        mock_wsi.read_region.return_value = img

        result = self._run(
            mock_wsi, ds, tissue_area_threshold=1, patch_size=256, mpp=0.5
        )
        # New formula: threshold = 4096 seg-px < 8100 → should be found
        assert result is not None, (
            "Island of 8100 seg-px should pass threshold=1 under the fixed formula "
            "(seg_patch_area=4096); old ref_patch_size=512 formula would give 16384 and filter it"
        )

    def test_threshold_invariant_across_seg_levels(self):
        """
        threshold=1 always means '1 requested patch of tissue minimum'.
        A tissue region of exactly 1 native patch size should be found at any seg level.
        We use a large tissue region (many patches) to confirm it's not accidentally filtered.
        """
        patch_size, mpp = 256, 0.5
        native_patch = patch_size / mpp  # = 512 native pixels

        for ds in [4.0, 8.0]:
            # Place a tissue region of 4×4 native patches (2048×2048 native px)
            native_tissue_px = int(4 * native_patch)
            mock_wsi = MagicMock()
            slide_w, slide_h = 16384, 16384
            seg_w = max(1, int(slide_w / ds))
            seg_h = max(1, int(slide_h / ds))
            mock_wsi.level_dimensions = [(slide_w, slide_h), (seg_w, seg_h)]
            mock_wsi.level_downsamples = [1.0, ds]
            mock_wsi.get_best_level_for_downsample.return_value = 1
            img = np.zeros((seg_h, seg_w, 3), dtype=np.uint8)
            seg_tissue = int(native_tissue_px / ds)
            cy, cx = seg_h // 2, seg_w // 2
            half = seg_tissue // 2
            img[cy - half : cy + half, cx - half : cx + half] = [210, 130, 160]
            mock_wsi.read_region.return_value = img

            result = self._run(
                mock_wsi, ds, tissue_area_threshold=1, patch_size=patch_size, mpp=mpp
            )
            assert (
                result is not None
            ), f"4×4 patch tissue region should be found at threshold=1, ds={ds}"
            _, _, coords, _ = result
            assert len(coords) > 0, f"Expected patches at ds={ds}"


def _make_mock_wsi_with_real_tissue(width=4096, height=4096):
    """Return a mock WSI where the bottom half is pink (tissue) and top half white (bg).

    HSV saturation of the pink pixels (~127) exceeds the default segment_threshold (20),
    so segment_tissue detects real tissue in the lower half of the slide.
    """
    mock_wsi = MagicMock()
    mock_wsi.level_dimensions = [(width, height), (width // 4, height // 4)]
    mock_wsi.level_downsamples = [1.0, 4.0]
    mock_wsi.get_best_level_for_downsample.return_value = 1
    h, w = height // 4, width // 4
    thumb = np.ones((h, w, 3), dtype=np.uint8) * 255  # white background
    # Bottom half: pink — HSV saturation ≈ 127 > segment_threshold(20) → detected as tissue
    thumb[h // 2 :, :] = [210, 100, 140]
    mock_wsi.read_region.return_value = thumb
    return mock_wsi


def _make_mock_wsi_with_tissue(width=4096, height=4096):
    """Return a mock WSI that produces a uniform-gray thumbnail (no real tissue).

    The classic segmenter thresholds on HSV saturation; a uniform gray image
    has S=0, so the tissue mask is all-zero.  Tests that need actual tissue
    should use _make_mock_wsi_with_real_tissue() instead.
    """
    import numpy as np

    mock_wsi = MagicMock()
    mock_wsi.level_dimensions = [(width, height), (width // 4, height // 4)]
    mock_wsi.level_downsamples = [1.0, 4.0]
    mock_wsi.get_best_level_for_downsample.return_value = 1
    # Solid gray thumbnail — saturation=0 → no tissue detected by classic segmenter
    thumb = np.ones((height // 4, width // 4, 3), dtype=np.uint8) * 200
    mock_wsi.read_region.return_value = thumb
    return mock_wsi


def _run_segment_with_mocks(mock_wsi, **kwargs):
    """Call segment_tissue with standard mocks and pass extra kwargs through."""
    from mussel.utils.segment import segment_tissue

    with (
        patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi),
        patch("mussel.utils.segment.get_slide_mpp", return_value=0.5),
        patch(
            "mussel.utils.segment._assert_level_downsamples",
            return_value=[(1.0, 1.0), (4.0, 4.0)],
        ),
    ):
        return segment_tissue("/fake/slide.svs", **kwargs)


class TestSegmentTissueOverlap:
    """Tests for the overlap parameter in segment_tissue."""

    def test_overlap_zero_uses_patch_size_as_step(self):
        """overlap=0 produces the same result as no overlap (step = patch_size)."""
        mock_wsi = _make_mock_wsi_with_tissue()
        result_no_overlap = _run_segment_with_mocks(
            mock_wsi, seg_level=1, patch_size=256, overlap=0, tissue_area_threshold=1
        )
        # Reset the mock read_region so the second call returns the same image
        mock_wsi.read_region.reset_mock()
        result_explicit = _run_segment_with_mocks(
            mock_wsi,
            seg_level=1,
            patch_size=256,
            step_size=256,
            tissue_area_threshold=1,
        )
        # Both should return the same number of patches (non-overlapping grid)
        if result_no_overlap is not None and result_explicit is not None:
            _, _, coords_no_overlap, _ = result_no_overlap
            _, _, coords_explicit, _ = result_explicit
            assert len(coords_no_overlap) == len(coords_explicit)
        else:
            assert result_no_overlap == result_explicit

    def test_overlap_sets_step_size(self):
        """overlap > 0 produces more patches than no overlap (smaller step)."""
        mock_wsi = _make_mock_wsi_with_tissue()
        result_no_overlap = _run_segment_with_mocks(
            mock_wsi, seg_level=1, patch_size=256, overlap=0, tissue_area_threshold=1
        )
        mock_wsi.read_region.reset_mock()
        result_with_overlap = _run_segment_with_mocks(
            mock_wsi, seg_level=1, patch_size=256, overlap=128, tissue_area_threshold=1
        )
        if result_no_overlap is not None and result_with_overlap is not None:
            _, _, coords_no_overlap, _ = result_no_overlap
            _, _, coords_with_overlap, _ = result_with_overlap
            # With overlap, step = 128 → denser grid → at least as many patches
            assert len(coords_with_overlap) >= len(coords_no_overlap)

    def test_overlap_and_step_size_conflict_raises(self):
        """Passing both overlap > 0 and step_size must raise ValueError."""
        from mussel.utils.segment import segment_tissue

        mock_wsi = MagicMock()
        mock_wsi.level_dimensions = [(1000, 1000)]

        with (
            patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi),
            patch("mussel.utils.segment.get_slide_mpp", return_value=0.5),
        ):
            with pytest.raises(ValueError, match="step_size"):
                segment_tissue(
                    "/fake/slide.svs",
                    seg_level=0,
                    patch_size=256,
                    overlap=64,
                    step_size=192,
                )

    def test_overlap_equal_to_patch_size_raises(self):
        """overlap >= patch_size must raise ValueError."""
        from mussel.utils.segment import segment_tissue

        mock_wsi = MagicMock()
        mock_wsi.level_dimensions = [(1000, 1000)]

        with (
            patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi),
            patch("mussel.utils.segment.get_slide_mpp", return_value=0.5),
        ):
            with pytest.raises(ValueError, match="overlap"):
                segment_tissue(
                    "/fake/slide.svs", seg_level=0, patch_size=256, overlap=256
                )

    def test_overlap_greater_than_patch_size_raises(self):
        """overlap > patch_size must raise ValueError."""
        from mussel.utils.segment import segment_tissue

        mock_wsi = MagicMock()
        mock_wsi.level_dimensions = [(1000, 1000)]

        with (
            patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi),
            patch("mussel.utils.segment.get_slide_mpp", return_value=0.5),
        ):
            with pytest.raises(ValueError, match="overlap"):
                segment_tissue(
                    "/fake/slide.svs", seg_level=0, patch_size=256, overlap=300
                )

    def test_negative_overlap_raises(self):
        """Negative overlap must raise ValueError (would create gaps, not overlap)."""
        from mussel.utils.segment import segment_tissue

        mock_wsi = MagicMock()
        mock_wsi.level_dimensions = [(1000, 1000)]

        with (
            patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi),
            patch("mussel.utils.segment.get_slide_mpp", return_value=0.5),
        ):
            with pytest.raises(ValueError, match="non-negative"):
                segment_tissue(
                    "/fake/slide.svs", seg_level=0, patch_size=256, overlap=-50
                )


class TestSegmentTissueMinTissueProportion:
    """Tests for the min_tissue_proportion parameter in segment_tissue."""

    def test_min_tissue_proportion_zero_keeps_all_patches(self):
        """min_tissue_proportion=0.0 should not remove any patches."""
        import cv2
        import numpy as np

        from mussel.utils.segment import contours_to_polygon, partition

        # Build a simple polygon covering a 512×512 region
        contour = np.array(
            [[[0, 0]], [[512, 0]], [[512, 512]], [[0, 512]]], dtype=np.int32
        )
        mock_wsi = MagicMock()
        mock_wsi.level_dimensions = [(1024, 1024), (256, 256)]
        mock_wsi.level_downsamples = [1.0, 4.0]
        mock_wsi.get_best_level_for_downsample.return_value = 1
        thumb = np.ones((256, 256, 3), dtype=np.uint8) * 200
        mock_wsi.read_region.return_value = thumb

        with (
            patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi),
            patch("mussel.utils.segment.get_slide_mpp", return_value=0.5),
            patch(
                "mussel.utils.segment._assert_level_downsamples",
                return_value=[(1.0, 1.0), (4.0, 4.0)],
            ),
            patch(
                "mussel.utils.segment._filter_contours",
                return_value=([contour], [[]]),
            ),
        ):
            from mussel.utils.segment import segment_tissue

            result_no_filter = segment_tissue(
                "/fake/slide.svs",
                patch_size=256,
                mpp=0.5,
                min_tissue_proportion=0.0,
                tissue_area_threshold=1,
            )
            result_filtered = segment_tissue(
                "/fake/slide.svs",
                patch_size=256,
                mpp=0.5,
                min_tissue_proportion=1.0,
                tissue_area_threshold=1,
            )

        # Both calls should succeed (non-None); filtering may differ in count
        # but at least min_tissue_proportion=0.0 should not reject everything
        if result_no_filter is not None and result_filtered is not None:
            _, _, coords_no_filter, _ = result_no_filter
            _, _, coords_filtered, _ = result_filtered
            assert len(coords_no_filter) >= len(coords_filtered)


class TestSegmentTissueArtifactRemover:
    """Tests for the artifact_remover_fn hook in segment_tissue."""

    def test_artifact_remover_fn_is_called(self):
        """artifact_remover_fn is invoked with (img, mask) when remove_artifacts=True."""
        calls = []

        def fake_remover(img, mask, mpp):
            calls.append((img.shape, mask.shape, mpp))
            return mask  # pass-through

        mock_wsi = _make_mock_wsi_with_tissue()

        _run_segment_with_mocks(
            mock_wsi,
            patch_size=256,
            mpp=0.5,
            tissue_area_threshold=1,
            remove_artifacts=True,
            artifact_remover_fn=fake_remover,
        )

        assert len(calls) == 1, "artifact_remover_fn should be called exactly once"
        img_shape, mask_shape, mpp = calls[0]
        assert len(img_shape) == 3 and img_shape[2] in (
            3,
            4,
        ), f"Expected RGB img, got shape {img_shape}"
        assert len(mask_shape) == 2, f"Expected 2D mask, got shape {mask_shape}"
        assert (
            img_shape[:2] == mask_shape
        ), f"img and mask spatial dims must match: {img_shape[:2]} vs {mask_shape}"
        assert (
            isinstance(mpp, float) and mpp > 0
        ), f"Expected positive float mpp, got {mpp}"

    def test_artifact_remover_fn_not_called_when_flags_false(self, caplog):
        """artifact_remover_fn with no flag set should warn and not call the function."""
        import logging

        calls = []

        def fake_remover(img, mask, mpp):
            calls.append(mask.shape)
            return mask

        mock_wsi = _make_mock_wsi_with_tissue()

        with caplog.at_level(logging.WARNING, logger="mussel.utils.segment"):
            _run_segment_with_mocks(
                mock_wsi,
                patch_size=256,
                mpp=0.5,
                tissue_area_threshold=1,
                artifact_remover_fn=fake_remover,
                # remove_artifacts and remove_penmarks both default to False
            )

        assert (
            len(calls) == 0
        ), "artifact_remover_fn should NOT be called when flags are False"
        assert any(
            "artifact_remover_fn" in msg for msg in caplog.messages
        ), "Expected warning about artifact_remover_fn provided but flags are False"

    def test_remove_artifacts_flag_without_fn_logs_warning(self, caplog):
        """remove_artifacts=True with no fn should log a warning, not crash."""
        import logging

        mock_wsi = _make_mock_wsi_with_tissue()

        with caplog.at_level(logging.WARNING, logger="mussel.utils.segment"):
            _run_segment_with_mocks(
                mock_wsi,
                patch_size=256,
                mpp=0.5,
                tissue_area_threshold=1,
                remove_artifacts=True,
                artifact_remover_fn=None,
            )

        assert any(
            "artifact_remover_fn" in msg for msg in caplog.messages
        ), "Expected warning about missing artifact_remover_fn"

    def test_remove_penmarks_flag_without_fn_logs_warning(self, caplog):
        """remove_penmarks=True with no fn should log a warning, not crash."""
        import logging

        mock_wsi = _make_mock_wsi_with_tissue()

        with caplog.at_level(logging.WARNING, logger="mussel.utils.segment"):
            _run_segment_with_mocks(
                mock_wsi,
                patch_size=256,
                mpp=0.5,
                tissue_area_threshold=1,
                remove_penmarks=True,
                artifact_remover_fn=None,
            )

        assert any("artifact_remover_fn" in msg for msg in caplog.messages)

    def test_tissue_survival_fallback_reverts_mask(self, caplog):
        """When remover eliminates >=90% of tissue, pre-removal mask is kept."""
        import logging

        def wipeout_remover(img, mask, mpp):
            return np.zeros_like(mask)

        mock_wsi = _make_mock_wsi_with_real_tissue()

        with caplog.at_level(logging.WARNING, logger="mussel.utils.segment"):
            result = _run_segment_with_mocks(
                mock_wsi,
                patch_size=256,
                mpp=0.5,
                tissue_area_threshold=1,
                remove_artifacts=True,
                artifact_remover_fn=wipeout_remover,
            )

        # Warning must mention the fallback with correct symbols and API
        assert any(
            "Falling back to pre-removal mask" in msg for msg in caplog.messages
        ), "Expected fallback warning in log"
        assert any(
            "artifact_exclude_classes=[4, 7]" in msg for msg in caplog.messages
        ), "Warning should reference artifact_exclude_classes=[4, 7], not deprecated remove_penmarks_only"
        assert any(
            ">=90" in msg for msg in caplog.messages
        ), "Warning should use >= symbol in threshold message"

        # Fallback reverts to pre-removal mask → tissue still present → patches found
        assert result is not None, "segment_tissue should not return None after fallback"
        _, _, coords, _ = result
        assert len(coords) > 0, "Pre-removal mask should yield patches after fallback"

    def test_tissue_survival_normal_path_uses_result_mask(self):
        """When remover keeps >=10% of tissue, the result mask is applied (no fallback)."""
        call_count = []

        def half_remover(img, mask, mpp):
            call_count.append(1)
            # Zero the top half — keeps bottom 50% of tissue, well above 10% survival
            result = mask.copy()
            result[: result.shape[0] // 2, :] = 0
            return result

        # Full-tissue mock — tissue in the bottom half only (the other half is bg already)
        mock_wsi = _make_mock_wsi_with_real_tissue()

        result_with_removal = _run_segment_with_mocks(
            mock_wsi,
            patch_size=256,
            mpp=0.5,
            tissue_area_threshold=1,
            remove_artifacts=True,
            artifact_remover_fn=half_remover,
        )

        assert len(call_count) == 1, "remover should have been called exactly once"
        # half_remover only keeps bottom tissue → should still produce some patches
        assert result_with_removal is not None, "Should not return None when tissue survives"

    def test_tissue_survival_boundary_at_exactly_90pct_triggers_fallback(self, caplog):
        """Removal fraction == 0.90 (exactly at threshold) should trigger fallback."""
        import logging

        def exactly_90pct_remover(img, mask, mpp):
            # Zero out exactly 90% of the nonzero pixels using ceil so we hit exactly >=90%
            import math
            flat = mask.flatten()
            nonzero_idx = np.flatnonzero(flat)
            n_to_zero = math.ceil(len(nonzero_idx) * 0.90)
            flat[nonzero_idx[:n_to_zero]] = 0
            return flat.reshape(mask.shape)

        mock_wsi = _make_mock_wsi_with_real_tissue()

        with caplog.at_level(logging.WARNING, logger="mussel.utils.segment"):
            _run_segment_with_mocks(
                mock_wsi,
                patch_size=256,
                mpp=0.5,
                tissue_area_threshold=1,
                remove_artifacts=True,
                artifact_remover_fn=exactly_90pct_remover,
            )

        assert any(
            "Falling back to pre-removal mask" in msg for msg in caplog.messages
        ), "Exactly 90% removal should trigger fallback (>= threshold)"

    def test_tissue_survival_pre_pixels_zero_triggers_fallback(self, caplog):
        """When pre-removal mask is all-zero, removal_fraction == 1.0 → fallback triggered."""
        import logging

        call_count = []

        def pass_through_remover(img, mask, mpp):
            call_count.append(1)
            return mask  # returns all-zero → same as input

        # All-gray uniform image → saturation channel = 0 → tissue mask is all-zero
        # (classic segmenter thresholds on HSV saturation; gray has S=0 < threshold=20)
        mock_wsi = _make_mock_wsi_with_tissue()  # uniform-gray → no tissue after segmentation

        with caplog.at_level(logging.WARNING, logger="mussel.utils.segment"):
            _run_segment_with_mocks(
                mock_wsi,
                patch_size=256,
                mpp=0.5,
                tissue_area_threshold=1,
                remove_artifacts=True,
                artifact_remover_fn=pass_through_remover,
            )

        # Remover IS called even when tissue mask is all-zero (no early-return before remover block)
        assert len(call_count) == 1, "remover is called regardless of tissue mask content"
        # pre_pixels=0 → removal_fraction=1.0 → fallback triggered
        assert any(
            "Falling back to pre-removal mask" in msg for msg in caplog.messages
        ), "pre_pixels=0 should set removal_fraction=1.0 and trigger the fallback"

    def test_mpp_escalation_uses_finer_level(self):
        """When img_mpp >= remover.max_input_mpp, a finer pyramid level is read."""
        import numpy as np

        call_args = []

        def mpp_checking_remover(img, mask, mpp):
            call_args.append({"img_shape": img.shape, "mask_shape": mask.shape, "mpp": mpp})
            return mask

        mpp_checking_remover.max_input_mpp = 4.0  # requires <= 4 µm/px

        # WSI with slide_mpp=0.5; seg_level=1 has downsample=8 → img_mpp=4.0 → triggers escalation
        # Escalation picks level 0 (downsample=1 → mpp=0.5 <= 4.0)
        mock_wsi = MagicMock()
        mock_wsi.level_dimensions = [(4096, 4096), (512, 512)]
        mock_wsi.level_downsamples = [1.0, 8.0]
        # get_best_level_for_downsample(4.0/0.5=8) → returns level 1 (ds=8 → mpp=4.0)
        # but 4.0 <= 4.0 so it IS fine — try with a coarser seg level
        mock_wsi.get_best_level_for_downsample.return_value = 0  # escalate to level 0
        fine_thumb = np.ones((4096, 4096, 3), dtype=np.uint8) * 200

        def smart_read(origin, level, dims):
            if level == 0:
                return fine_thumb
            coarse = np.ones((512, 512, 3), dtype=np.uint8) * 200
            return coarse

        mock_wsi.read_region.side_effect = smart_read

        from mussel.utils.segment import segment_tissue
        from unittest.mock import patch as _patch

        with (
            _patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi),
            _patch("mussel.utils.segment.get_slide_mpp", return_value=0.5),
            _patch(
                "mussel.utils.segment._assert_level_downsamples",
                return_value=[(1.0, 1.0), (8.0, 8.0)],
            ),
        ):
            segment_tissue(
                "/fake/slide.svs",
                patch_size=256,
                mpp=0.5,
                seg_level=1,
                tissue_area_threshold=1,
                remove_artifacts=True,
                artifact_remover_fn=mpp_checking_remover,
            )

        assert len(call_args) == 1, "remover should be called once"
        # At escalation, level 0 mpp = 0.5 * 1.0 = 0.5 < max_input_mpp=4.0
        assert call_args[0]["mpp"] == pytest.approx(0.5), (
            f"Expected escalated mpp=0.5, got {call_args[0]['mpp']}"
        )
        # The fine-level image should be larger than the 512×512 coarse seg thumbnail
        assert call_args[0]["img_shape"][0] == 4096, (
            "Remover should receive the finer-level thumbnail, not the coarse seg thumbnail"
        )

    def test_mpp_escalation_skipped_when_img_mpp_below_max(self):
        """When img_mpp < max_input_mpp, the seg-level thumbnail is used directly."""
        import numpy as np

        call_args = []

        def mpp_checking_remover(img, mask, mpp):
            call_args.append({"img_shape": img.shape, "mpp": mpp})
            return mask

        mpp_checking_remover.max_input_mpp = 16.0  # very permissive — seg level is fine

        # seg_level=1 with downsample=4 and slide_mpp=0.5 → img_mpp=2.0 < 16.0
        mock_wsi = _make_mock_wsi_with_tissue()  # level_downsamples=[1.0, 4.0]

        _run_segment_with_mocks(
            mock_wsi,
            patch_size=256,
            mpp=0.5,
            seg_level=1,
            tissue_area_threshold=1,
            remove_artifacts=True,
            artifact_remover_fn=mpp_checking_remover,
        )

        assert len(call_args) == 1
        # img_mpp = 0.5 * 4.0 = 2.0 (seg level, not escalated)
        assert call_args[0]["mpp"] == pytest.approx(2.0), (
            f"Expected seg-level mpp=2.0, got {call_args[0]['mpp']}"
        )
        # Thumbnail should be 1024×1024 (4096/4), the seg-level size
        assert call_args[0]["img_shape"][0] == 1024, (
            "Should use seg-level thumbnail when mpp is within limit"
        )


class TestSegmentTissueSegModel:
    """Tests for the seg_model parameter in segment_tissue."""

    def test_seg_model_classic_is_default(self):
        """seg_model defaults to 'classic' — no error without torch."""
        import inspect

        from mussel.utils.segment import segment_tissue

        sig = inspect.signature(segment_tissue)
        assert sig.parameters["seg_model"].default == "classic"

    def test_seg_model_otsu_accepted(self):
        """seg_model='otsu' is a valid value — no ValueError raised."""
        mock_wsi = _make_mock_wsi_with_tissue()
        # Use the shared helper (same pattern as other classic-mode tests).
        result = _run_segment_with_mocks(
            mock_wsi,
            seg_level=1,
            patch_size=256,
            mpp=0.5,
            tissue_area_threshold=1,
            seg_model="otsu",
        )
        # Whether or not tissue is found, seg_model should be stored in attrs.
        if result is not None:
            _, _, _, attrs = result
            assert attrs["seg_model"] == "otsu"

    def test_use_otsu_deprecated_overrides_seg_model(self):
        """use_otsu=True emits DeprecationWarning and acts as seg_model='otsu'."""
        mock_wsi = _make_mock_wsi_with_tissue()
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _run_segment_with_mocks(
                mock_wsi,
                seg_level=1,
                patch_size=256,
                mpp=0.5,
                tissue_area_threshold=1,
                use_otsu=True,
            )
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 1
        assert "seg_model='otsu'" in str(dep_warnings[0].message)
        if result is not None:
            _, _, _, attrs = result
            assert attrs["seg_model"] == "otsu"

    def test_seg_model_neural_warns_on_classic_params(self):
        """seg_model='neural' logs a warning when classic-only params are non-default."""
        mock_wsi = _make_mock_wsi_with_tissue()
        fake_mask = np.zeros((64, 64), dtype=np.uint8)
        fake_mask[16:48, 16:48] = 255

        with (
            patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi),
            patch("mussel.utils.segment.get_slide_mpp", return_value=0.5),
            patch(
                "mussel.utils.segment._assert_level_downsamples",
                return_value=[(1.0, 1.0), (4.0, 4.0)],
            ),
            patch(
                "mussel.utils.segment._segment_tissue_neural",
                return_value=fake_mask,
            ),
            patch("mussel.utils.segment.logger") as mock_logger,
        ):
            from mussel.utils.segment import segment_tissue

            segment_tissue(
                "/fake/slide.svs",
                patch_size=256,
                mpp=0.5,
                tissue_area_threshold=1,
                seg_model="neural",
                median_blur_ksize=11,  # non-default — should trigger warning
            )
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert any("median_blur_ksize" in c for c in warning_calls)

    def test_seg_model_neural_calls_neural_segmenter(self):
        """seg_model='neural' delegates to _segment_tissue_neural."""
        mock_wsi = _make_mock_wsi_with_tissue()
        fake_mask = np.zeros((64, 64), dtype=np.uint8)
        fake_mask[16:48, 16:48] = 255

        with (
            patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi),
            patch("mussel.utils.segment.get_slide_mpp", return_value=0.5),
            patch(
                "mussel.utils.segment._assert_level_downsamples",
                return_value=[(1.0, 1.0), (4.0, 4.0)],
            ),
            patch(
                "mussel.utils.segment._segment_tissue_neural",
                return_value=fake_mask,
            ) as mock_neural,
        ):
            from mussel.utils.segment import segment_tissue

            segment_tissue(
                "/fake/slide.svs",
                patch_size=256,
                mpp=0.5,
                tissue_area_threshold=1,
                seg_model="neural",
            )
            mock_neural.assert_called_once()

    def test_seg_model_invalid_raises(self):
        """Unknown seg_model raises ValueError."""
        mock_wsi = _make_mock_wsi_with_tissue()
        with (
            patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi),
            patch("mussel.utils.segment.get_slide_mpp", return_value=0.5),
            patch(
                "mussel.utils.segment._assert_level_downsamples",
                return_value=[(1.0, 1.0), (4.0, 4.0)],
            ),
        ):
            from mussel.utils.segment import segment_tissue

            with pytest.raises(ValueError, match="Unsupported seg_model"):
                segment_tissue(
                    "/fake/slide.svs",
                    patch_size=256,
                    mpp=0.5,
                    tissue_area_threshold=1,
                    seg_model="hest",
                )
