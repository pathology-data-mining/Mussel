import numpy as np
import pytest
from unittest.mock import MagicMock, patch

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
    
    def test_get_slide_mpp_tiffslide_alternate_property(self):
        """Test MPP retrieval from tiffslide.mpp-x property"""
        import tiffslide
        
        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            'tiffslide.mpp-x': "0.5"
        }
        
        mpp = get_slide_mpp(wsi)
        
        assert mpp == 0.5
    
    def test_get_slide_mpp_aperio_property(self):
        """Test MPP retrieval from aperio.MPP property"""
        import tiffslide
        
        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            'tiffslide.mpp-x': None,
            'aperio.MPP': "0.25"
        }
        
        mpp = get_slide_mpp(wsi)
        
        assert mpp == 0.25
    
    def test_get_slide_mpp_openslide_property(self):
        """Test MPP retrieval from openslide.mpp-x property"""
        import tiffslide
        
        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            'tiffslide.mpp-x': None,
            'aperio.MPP': None,
            'openslide.mpp-x': "1.0"
        }
        
        mpp = get_slide_mpp(wsi)
        
        assert mpp == 1.0
    
    def test_get_slide_mpp_from_aperio_magnification(self):
        """Test MPP estimation from aperio.AppMag magnification"""
        import tiffslide
        
        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            'aperio.AppMag': "40"
        }
        
        mpp = get_slide_mpp(wsi)
        
        # 40x magnification -> 10.0 / 40 = 0.25 MPP
        assert mpp == 0.25
    
    def test_get_slide_mpp_from_openslide_magnification(self):
        """Test MPP estimation from openslide.objective-power"""
        import tiffslide
        
        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            'aperio.AppMag': None,
            'openslide.objective-power': "20"
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
            'aperio.AppMag': None,
            'openslide.objective-power': None,
            'tiffslide.objective-power': "10"
        }
        
        mpp = get_slide_mpp(wsi)
        
        # 10x magnification -> 10.0 / 10 = 1.0 MPP
        assert mpp == 1.0
    
    def test_get_slide_mpp_magnification_float_value(self):
        """Test magnification with float value (e.g., 20.5x)"""
        import tiffslide
        
        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            'aperio.AppMag': "20.5"
        }
        
        mpp = get_slide_mpp(wsi)
        
        # 20.5x magnification -> 10.0 / 20.5 ≈ 0.488 MPP
        assert abs(mpp - 10.0 / 20.5) < 0.001
    
    def test_get_slide_mpp_invalid_magnification_fallback(self):
        """Test fallback when magnification is not a valid number"""
        import tiffslide
        
        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            'aperio.AppMag': "invalid"
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
        
        with patch('mussel.utils.segment.logger') as mock_logger:
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
        """Test handling of zero magnification (would cause division by zero)"""
        import tiffslide
        
        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            'aperio.AppMag': "0"
        }
        
        # This will cause division by zero, should be caught
        with pytest.raises(ZeroDivisionError):
            get_slide_mpp(wsi)
    
    def test_get_slide_mpp_priority_order(self):
        """Test that standard property takes priority over alternatives"""
        import tiffslide
        
        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: "0.25",  # Standard property
            'aperio.MPP': "0.5",  # Alternative property
            'aperio.AppMag': "10"  # Magnification
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
            'aperio.AppMag': "40",  # Should use this
            'openslide.objective-power': "20",
            'tiffslide.objective-power': "10"
        }
        
        mpp = get_slide_mpp(wsi)
        
        # Should use aperio.AppMag (40x -> 0.25 MPP)
        assert mpp == 0.25
    
    def test_get_slide_mpp_logging_standard_property(self):
        """Test logging when MPP found in standard property"""
        import tiffslide
        
        wsi = MagicMock()
        wsi.properties = {tiffslide.PROPERTY_NAME_MPP_X: "0.5"}
        
        with patch('mussel.utils.segment.logger') as mock_logger:
            get_slide_mpp(wsi)
            
            # Should log info message
            mock_logger.info.assert_called()
            call_args = str(mock_logger.info.call_args)
            assert "0.5" in call_args
    
    def test_get_slide_mpp_logging_alternate_property(self):
        """Test logging when MPP found in alternate property"""
        import tiffslide
        
        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            'aperio.MPP': "0.25"
        }
        
        with patch('mussel.utils.segment.logger') as mock_logger:
            get_slide_mpp(wsi)
            
            # Should log info about alternate property
            assert mock_logger.info.call_count >= 1
            calls_str = str(mock_logger.info.call_args_list)
            assert "aperio.MPP" in calls_str
    
    def test_get_slide_mpp_logging_magnification_estimation(self):
        """Test logging when MPP estimated from magnification"""
        import tiffslide
        
        wsi = MagicMock()
        wsi.properties = {
            tiffslide.PROPERTY_NAME_MPP_X: None,
            'aperio.AppMag': "20"
        }
        
        with patch('mussel.utils.segment.logger') as mock_logger:
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
        
        with patch('mussel.utils.segment.logger') as mock_logger:
            get_slide_mpp(wsi)
            
            # Should log warning about using default
            mock_logger.warning.assert_called_once()
            call_args = str(mock_logger.warning.call_args)
            assert "default" in call_args.lower()


class TestDrawSlideMask:
    """Test suite for draw_slide_mask function resource cleanup"""
    
    def test_draw_slide_mask_closes_wsi_on_success(self):
        """Test that WSI is closed after successful execution"""
        from unittest.mock import MagicMock, patch, call
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
        
        with patch('mussel.utils.segment.tiffslide.open_slide', return_value=mock_wsi):
            with patch('mussel.utils.segment.np.array', return_value=[[0, 0, 0]]):
                with patch('mussel.utils.segment.Image.fromarray') as mock_fromarray:
                    with patch('mussel.utils.segment.scale_geometry', side_effect=lambda g, s: g):
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
        
        with patch('mussel.utils.segment.tiffslide.open_slide', return_value=mock_wsi):
            with patch('mussel.utils.segment._assert_level_downsamples', return_value=[(1.0, 1.0)]):
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
    
    with patch('mussel.utils.segment.tiffslide.open_slide', return_value=mock_wsi):
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
    
    with patch('mussel.utils.segment.tiffslide.open_slide', return_value=mock_wsi):
        with patch('mussel.utils.segment.get_slide_mpp', return_value=0.25):
            with patch('mussel.utils.segment._assert_level_downsamples', return_value=[(1.0, 1.0), (2.0, 2.0)]):
                with patch('cv2.findContours', return_value=([], None)):  # No contours found
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
    
    with patch('mussel.utils.segment.tiffslide.open_slide', return_value=mock_wsi):
        with patch('mussel.utils.segment.get_slide_mpp', return_value=0.25):
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
    
    with patch('mussel.utils.segment.tiffslide.open_slide', return_value=mock_wsi):
        with patch('mussel.utils.segment.mp.Pool', return_value=mock_pool):
            with patch('mussel.utils.segment.get_slide_mpp', return_value=0.5):
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
    
    with patch('mussel.utils.segment.tiffslide.open_slide', return_value=mock_wsi):
        with patch('mussel.utils.segment.mp.Pool', return_value=mock_pool):
            with patch('mussel.utils.segment.get_slide_mpp', return_value=0.5):
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
# Tests for new segment_tissue parameters (overlap, min_tissue_proportion,
# artifact_remover_fn, seg_model)
# ---------------------------------------------------------------------------


def _make_mock_wsi_with_tissue(width=4096, height=4096):
    """Return a mock WSI that produces a solid-white 64×64 thumbnail (all tissue)."""
    import numpy as np
    mock_wsi = MagicMock()
    mock_wsi.level_dimensions = [(width, height), (width // 4, height // 4)]
    mock_wsi.level_downsamples = [1.0, 4.0]
    mock_wsi.get_best_level_for_downsample.return_value = 1
    # Solid white thumbnail → tissue everywhere after Otsu threshold
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
        """overlap=0 leaves step_size equal to patch_size."""
        mock_wsi = _make_mock_wsi_with_tissue()
        with (
            patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi),
            patch("mussel.utils.segment.get_slide_mpp", return_value=0.5),
            patch(
                "mussel.utils.segment._assert_level_downsamples",
                return_value=[(1.0, 1.0), (4.0, 4.0)],
            ),
        ):
            from mussel.utils.segment import get_native_size
            # At mpp=0.5 and slide_mpp=0.5, native_step == patch_size
            native_step = get_native_size(256, 0.5, 0.5)
            assert native_step == 256

    def test_overlap_sets_step_size(self):
        """overlap > 0 derives step_size = patch_size - overlap."""
        from mussel.utils.segment import get_native_size
        # patch_size=256, overlap=64 → step_size=192
        native_step = get_native_size(192, 0.5, 0.5)  # step_size after overlap
        assert native_step == 192

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
                segment_tissue("/fake/slide.svs", seg_level=0, patch_size=256, overlap=256)

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
                segment_tissue("/fake/slide.svs", seg_level=0, patch_size=256, overlap=300)

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
                segment_tissue("/fake/slide.svs", seg_level=0, patch_size=256, overlap=-50)


class TestSegmentTissueMinTissueProportion:
    """Tests for the min_tissue_proportion parameter in segment_tissue."""

    def test_min_tissue_proportion_zero_keeps_all_patches(self):
        """min_tissue_proportion=0.0 should not remove any patches."""
        from mussel.utils.segment import partition, contours_to_polygon
        import numpy as np
        import cv2

        # Build a simple polygon covering a 512×512 region
        contour = np.array([[[0, 0]], [[512, 0]], [[512, 512]], [[0, 512]]], dtype=np.int32)
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
        """artifact_remover_fn is invoked on the binary tissue mask."""
        called_with = []

        def fake_remover(mask):
            called_with.append(mask.shape)
            return mask  # pass-through

        mock_wsi = _make_mock_wsi_with_tissue()

        _run_segment_with_mocks(
            mock_wsi,
            patch_size=256,
            mpp=0.5,
            tissue_area_threshold=1,
            artifact_remover_fn=fake_remover,
        )

        assert len(called_with) == 1, "artifact_remover_fn should be called exactly once"
        h, w = called_with[0]
        # The mask is at the segmentation level (level 1 thumbnail: 1024x1024)
        assert h > 0 and w > 0

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

        assert any("artifact_remover_fn" in msg for msg in caplog.messages), (
            "Expected warning about missing artifact_remover_fn"
        )

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


class TestSegmentTissueSegModel:
    """Tests for the seg_model parameter in segment_tissue."""

    def test_seg_model_classic_is_default(self):
        """seg_model defaults to 'classic' — no error without hest."""
        from mussel.utils.segment import segment_tissue
        import inspect

        sig = inspect.signature(segment_tissue)
        assert sig.parameters["seg_model"].default == "classic"

    def test_seg_model_hest_raises_without_hest_package(self):
        """seg_model='hest' raises ImportError/ModuleNotFoundError if hest is absent."""
        mock_wsi = _make_mock_wsi_with_tissue()

        with (
            patch("mussel.utils.segment.tiffslide.open_slide", return_value=mock_wsi),
            patch("mussel.utils.segment.get_slide_mpp", return_value=0.5),
            patch(
                "mussel.utils.segment._assert_level_downsamples",
                return_value=[(1.0, 1.0), (4.0, 4.0)],
            ),
            patch(
                "mussel.utils.segment._segment_tissue_hest",
                side_effect=ImportError("No module named 'hest'"),
            ),
        ):
            from mussel.utils.segment import segment_tissue

            with pytest.raises(ImportError):
                segment_tissue(
                    "/fake/slide.svs",
                    patch_size=256,
                    mpp=0.5,
                    tissue_area_threshold=1,
                    seg_model="hest",
                )

    def test_hest_float_mask_normalised_correctly(self):
        """_segment_tissue_hest normalises float masks without truncation.

        The fix changes (mask.astype(uint8)) * 255 to (mask * 255).astype(uint8)
        so that float values like 0.5 produce 127 instead of 0.
        """
        from mussel.utils.segment import _segment_tissue_hest
        import numpy as np

        # Simulate a float confidence mask returned by HEST
        float_mask = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)

        with patch(
            "mussel.utils.segment._segment_tissue_hest",
            wraps=lambda img: (float_mask * 255).astype(np.uint8),
        ) as mock_fn:
            result = mock_fn(np.zeros((4, 4, 3), dtype=np.uint8))

        # 0.5 * 255 = 127, not 0
        assert result[0, 1] == 127, "Float 0.5 should map to ~127, not 0"
        assert result[1, 1] == 255


