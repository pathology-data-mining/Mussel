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
