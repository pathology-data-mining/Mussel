"""Tests for model_predownload module."""
import os
import subprocess
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


def test_run_save_model_with_direct_command():
    """Test run_save_model uses direct command when save_model is in PATH."""
    from model_predownload import run_save_model
    
    with patch('shutil.which') as mock_which:
        with patch('subprocess.run') as mock_run:
            # Mock that save_model is available directly
            def which_side_effect(cmd):
                if cmd == 'save_model':
                    return '/usr/bin/save_model'
                return None
            
            mock_which.side_effect = which_side_effect
            mock_run.return_value = MagicMock(returncode=0)
            
            result = run_save_model('VIRCHOW', '/tmp/test.pth')
            
            # Should call save_model directly
            assert result is True
            assert mock_run.called
            first_call = mock_run.call_args_list[0]
            cmd = first_call[0][0]
            assert cmd[0] == 'save_model'
            assert 'model_type=VIRCHOW' in cmd


def test_run_save_model_with_uv_and_virtual_env():
    """Test run_save_model uses uv run when VIRTUAL_ENV is set."""
    from model_predownload import run_save_model
    
    with patch('shutil.which') as mock_which:
        with patch('subprocess.run') as mock_run:
            with patch.dict(os.environ, {'VIRTUAL_ENV': '/path/to/venv'}):
                # Mock that only uv is available, not save_model
                def which_side_effect(cmd):
                    if cmd == 'uv':
                        return '/usr/bin/uv'
                    return None
                
                mock_which.side_effect = which_side_effect
                mock_run.return_value = MagicMock(returncode=0)
                
                result = run_save_model('VIRCHOW', '/tmp/test.pth')
                
                # Should try uv run
                assert result is True
                assert mock_run.called
                first_call = mock_run.call_args_list[0]
                cmd = first_call[0][0]
                assert cmd[0:2] == ['uv', 'run']


def test_run_save_model_without_virtual_env_skips_uv():
    """Test run_save_model doesn't use uv run without VIRTUAL_ENV (SLURM scenario)."""
    from model_predownload import run_save_model
    
    with patch('shutil.which') as mock_which:
        with patch('subprocess.run') as mock_run:
            with patch.dict(os.environ, {}, clear=True):
                # Mock that only uv is available, not save_model
                # But no VIRTUAL_ENV set (typical SLURM submission node)
                def which_side_effect(cmd):
                    if cmd == 'uv':
                        return '/usr/bin/uv'
                    return None
                
                mock_which.side_effect = which_side_effect
                mock_run.return_value = MagicMock(returncode=0)
                
                result = run_save_model('VIRCHOW', '/tmp/test.pth')
                
                # Should NOT use uv run, should use python -m fallback
                assert result is True
                assert mock_run.called
                first_call = mock_run.call_args_list[0]
                cmd = first_call[0][0]
                # Should be python -m mussel.cli.save_model
                assert '-m' in cmd
                assert 'mussel.cli.save_model' in cmd


def test_run_save_model_fallback_chain():
    """Test run_save_model tries multiple commands on failure."""
    from model_predownload import run_save_model
    
    with patch('shutil.which') as mock_which:
        with patch('subprocess.run') as mock_run:
            # Mock that save_model is available
            mock_which.return_value = '/usr/bin/save_model'
            
            # First call fails, second succeeds
            mock_run.side_effect = [
                subprocess.CalledProcessError(1, 'save_model', stderr='error'),
                MagicMock(returncode=0)
            ]
            
            result = run_save_model('VIRCHOW', '/tmp/test.pth')
            
            # Should have tried at least twice
            assert result is True
            assert mock_run.call_count >= 2
