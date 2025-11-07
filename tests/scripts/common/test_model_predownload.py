"""Tests for model_predownload module."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts/common to path
scripts_common = str(Path(__file__).parent.parent.parent.parent / "scripts" / "common")
sys.path.insert(0, scripts_common)


def test_can_model_be_saved_with_path():
    """Test can_model_be_saved returns True for models with non-empty paths."""
    from model_predownload import can_model_be_saved
    
    # Create a mock ModelType enum with a model that has a path
    mock_model_enum = MagicMock()
    mock_model_enum.path = "hf-hub:some/model"
    
    mock_ModelType = MagicMock()
    mock_ModelType.__getitem__ = MagicMock(return_value=mock_model_enum)
    
    with patch.dict('sys.modules', {'mussel.models.model_factory': MagicMock(ModelType=mock_ModelType)}):
        result = can_model_be_saved("VIRCHOW")
        assert result is True


def test_can_model_be_saved_without_path():
    """Test can_model_be_saved returns False for models with empty paths."""
    from model_predownload import can_model_be_saved
    
    # Create a mock ModelType enum with a model that has an empty path
    mock_model_enum = MagicMock()
    mock_model_enum.path = ""
    
    mock_ModelType = MagicMock()
    mock_ModelType.__getitem__ = MagicMock(return_value=mock_model_enum)
    
    with patch.dict('sys.modules', {'mussel.models.model_factory': MagicMock(ModelType=mock_ModelType)}):
        result = can_model_be_saved("CTRANSPATH")
        assert result is False


def test_can_model_be_saved_with_whitespace_path():
    """Test can_model_be_saved returns False for models with whitespace-only paths."""
    from model_predownload import can_model_be_saved
    
    # Create a mock ModelType enum with a model that has a whitespace path
    mock_model_enum = MagicMock()
    mock_model_enum.path = "   "
    
    mock_ModelType = MagicMock()
    mock_ModelType.__getitem__ = MagicMock(return_value=mock_model_enum)
    
    with patch.dict('sys.modules', {'mussel.models.model_factory': MagicMock(ModelType=mock_ModelType)}):
        result = can_model_be_saved("SOMEMODEL")
        assert result is False


def test_can_model_be_saved_with_none_path():
    """Test can_model_be_saved returns False for models with None paths."""
    from model_predownload import can_model_be_saved
    
    # Create a mock ModelType enum with a model that has None path
    mock_model_enum = MagicMock()
    mock_model_enum.path = None
    
    mock_ModelType = MagicMock()
    mock_ModelType.__getitem__ = MagicMock(return_value=mock_model_enum)
    
    with patch.dict('sys.modules', {'mussel.models.model_factory': MagicMock(ModelType=mock_ModelType)}):
        result = can_model_be_saved("SOMEMODEL")
        assert result is False


def test_can_model_be_saved_unknown_model():
    """Test can_model_be_saved returns False for unknown model types."""
    from model_predownload import can_model_be_saved
    
    # Create a mock ModelType that raises KeyError for unknown models
    mock_ModelType = MagicMock()
    mock_ModelType.__getitem__ = MagicMock(side_effect=KeyError("Unknown model"))
    
    with patch.dict('sys.modules', {'mussel.models.model_factory': MagicMock(ModelType=mock_ModelType)}):
        result = can_model_be_saved("UNKNOWN_MODEL")
        assert result is False


def test_can_model_be_saved_import_error():
    """Test can_model_be_saved returns True (fallback) when import fails."""
    from model_predownload import can_model_be_saved
    
    # Simulate import error by removing the module from sys.modules
    with patch.dict('sys.modules', {'mussel.models.model_factory': None}):
        # Force reimport to trigger ImportError
        with patch('builtins.__import__', side_effect=ImportError("No module")):
            result = can_model_be_saved("SOMEMODEL")
            # Fallback behavior is to return True
            assert result is True


def test_pre_download_models_skips_unsavable():
    """Test pre_download_models skips models that cannot be saved."""
    from model_predownload import pre_download_models
    
    # Mock can_model_be_saved to return False for CTRANSPATH
    with patch('model_predownload.can_model_be_saved') as mock_can_save:
        mock_can_save.side_effect = lambda model: model != "CTRANSPATH"
        
        with patch('model_predownload.run_save_model') as mock_run_save:
            with patch('model_predownload.Path') as mock_path:
                mock_path.return_value.mkdir = MagicMock()
                
                result = pre_download_models(
                    model_types=["CTRANSPATH", "VIRCHOW"],
                    cache_dir="/tmp/cache"
                )
                
                # CTRANSPATH should be skipped, so run_save_model should only be called for VIRCHOW
                assert "CTRANSPATH" not in result
                # run_save_model should be called once (for VIRCHOW, not CTRANSPATH)
                # Note: it won't be called if the file already exists
                assert mock_can_save.call_count == 2
