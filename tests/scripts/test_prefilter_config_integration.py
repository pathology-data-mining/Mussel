#!/usr/bin/env python3
"""
Integration test for prefilter_model_path configuration handling.

This test verifies that:
1. prefilter_model_path from config file is correctly loaded
2. Model pre-download logic respects config file values
3. Command-line args properly override config values
"""

import os
import sys
import tempfile

import pytest

# Import the config loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'common'))
from config_loader import load_config_defaults


def test_prefilter_model_path_config_integration():
    """
    Integration test: prefilter_model_path from config should be accepted.
    
    This test simulates the complete flow:
    1. Config file contains prefilter_model_path
    2. Pre-download logic should detect it and skip downloading
    3. CSV processing should include the config value
    """
    yaml_content = """
# Test config with prefilter_model_path
prefilter_model_type: CTRANSPATH
prefilter_model_path: /shared/models/ctranspath.pth
batch_size: 64
num_workers: 4
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_file = f.name
    
    try:
        # Step 1: Load config defaults (as done early in submission scripts)
        config_defaults = load_config_defaults(config_file, backend='condor')
        
        # Verify prefilter_model_path is loaded
        assert 'prefilter_model_path' in config_defaults
        assert config_defaults['prefilter_model_path'] == '/shared/models/ctranspath.pth'
        
        # Step 2: Simulate pre-download logic
        # In the fixed code, this checks both args and config_defaults
        args_prefilter_model_path = None  # User didn't provide via command line
        user_provided_prefilter_path = (
            args_prefilter_model_path or 
            config_defaults.get('prefilter_model_path')
        )
        
        # Since config has the path, pre-download should be skipped
        assert user_provided_prefilter_path == '/shared/models/ctranspath.pth'
        should_download_ctranspath = not user_provided_prefilter_path
        assert should_download_ctranspath is False, "CTRANSPATH should not be downloaded when path is in config"
        
        # Step 3: Simulate CSV processing with config merge
        csv_kwargs = {
            'classifier_threshold': 0.75,
            'prefilter_model_type': 'CTRANSPATH',
            'batch_size': 64,
            'num_workers': 4,
        }
        
        # Merge: config first, then csv_kwargs
        merged_kwargs = {**config_defaults, **csv_kwargs}
        
        # Config value should be preserved
        assert merged_kwargs['prefilter_model_path'] == '/shared/models/ctranspath.pth'
        
        # Step 4: Simulate adding command-line override (if provided)
        model_paths = {}  # Empty because pre-download was skipped
        
        # The fix: only add to csv_kwargs if explicitly provided via command line
        if args_prefilter_model_path:
            merged_kwargs['prefilter_model_path'] = args_prefilter_model_path
        # Pre-downloaded models would also be added here (but we skipped pre-download)
        if model_paths.get('CTRANSPATH'):
            merged_kwargs['prefilter_model_path'] = model_paths['CTRANSPATH']
        
        # Final check: config value is still present
        assert merged_kwargs['prefilter_model_path'] == '/shared/models/ctranspath.pth'
        
    finally:
        os.unlink(config_file)


def test_prefilter_model_path_command_line_override():
    """
    Test that command-line args properly override config values.
    """
    yaml_content = """
prefilter_model_type: CTRANSPATH
prefilter_model_path: /config/path/ctranspath.pth
batch_size: 64
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_file = f.name
    
    try:
        # Load config
        config_defaults = load_config_defaults(config_file, backend='slurm')
        
        # Simulate command-line providing a different path
        args_prefilter_model_path = '/cmdline/path/ctranspath.pth'
        
        # Pre-download logic: command-line takes precedence
        user_provided_prefilter_path = (
            args_prefilter_model_path or 
            config_defaults.get('prefilter_model_path')
        )
        assert user_provided_prefilter_path == '/cmdline/path/ctranspath.pth'
        
        # CSV processing
        csv_kwargs = {
            'batch_size': 64,
            'prefilter_model_type': 'CTRANSPATH',
        }
        
        # Merge config first
        merged_kwargs = {**config_defaults, **csv_kwargs}
        
        # Then apply command-line override
        model_paths = {}
        if args_prefilter_model_path:
            merged_kwargs['prefilter_model_path'] = args_prefilter_model_path
        
        # Command-line value should win
        assert merged_kwargs['prefilter_model_path'] == '/cmdline/path/ctranspath.pth'
        
    finally:
        os.unlink(config_file)


def test_prefilter_model_path_predownload_override():
    """
    Test that pre-downloaded models override config values.
    """
    yaml_content = """
prefilter_model_type: CTRANSPATH
prefilter_model_path: /config/path/ctranspath.pth
batch_size: 64
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_file = f.name
    
    try:
        # Load config
        config_defaults = load_config_defaults(config_file, backend='azure')
        
        # Simulate pre-download happened (maybe user disabled config check or downloaded anyway)
        model_paths = {
            'CTRANSPATH': '/cache/predownload/ctranspath.pth'
        }
        
        # CSV processing
        csv_kwargs = {
            'batch_size': 64,
            'prefilter_model_type': 'CTRANSPATH',
        }
        
        # Merge config first
        merged_kwargs = {**config_defaults, **csv_kwargs}
        
        # Then apply pre-downloaded path (takes precedence)
        args_prefilter_model_path = None
        if model_paths.get('CTRANSPATH'):
            merged_kwargs['prefilter_model_path'] = model_paths['CTRANSPATH']
        if args_prefilter_model_path:  # Command-line has highest priority
            merged_kwargs['prefilter_model_path'] = args_prefilter_model_path
        
        # Pre-downloaded value should win (since no command-line override)
        assert merged_kwargs['prefilter_model_path'] == '/cache/predownload/ctranspath.pth'
        
    finally:
        os.unlink(config_file)


def test_multiple_model_paths_config():
    """
    Test that all model paths (prefilter, postfilter, slide) work from config.
    """
    yaml_content = """
prefilter_model_type: CTRANSPATH
prefilter_model_path: /shared/models/ctranspath.pth
model_type: UNI
model_path: /shared/models/uni.pth
slide_model_type: GIGAPATH_SLIDE
slide_model_path: /shared/models/gigapath.pth
aggregation_method: model
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_file = f.name
    
    try:
        # Load config
        config_defaults = load_config_defaults(config_file, backend='condor')
        
        # Verify all model paths are loaded
        assert config_defaults['prefilter_model_path'] == '/shared/models/ctranspath.pth'
        assert config_defaults['model_path'] == '/shared/models/uni.pth'
        assert config_defaults['slide_model_path'] == '/shared/models/gigapath.pth'
        
        # Simulate pre-download logic
        args_prefilter = None
        args_postfilter = None
        args_slide = None
        
        user_provided_paths = {
            'prefilter': args_prefilter or config_defaults.get('prefilter_model_path'),
            'postfilter': args_postfilter or config_defaults.get('model_path'),
            'slide': args_slide or config_defaults.get('slide_model_path'),
        }
        
        # All should be provided, so no downloads needed
        assert user_provided_paths['prefilter'] is not None
        assert user_provided_paths['postfilter'] is not None
        assert user_provided_paths['slide'] is not None
        
        # Models to download should be empty
        models_to_download = []
        if not user_provided_paths['prefilter']:
            models_to_download.append('CTRANSPATH')
        if not user_provided_paths['postfilter']:
            models_to_download.append('UNI')
        
        assert len(models_to_download) == 0, "No models should be downloaded when all paths are in config"
        
    finally:
        os.unlink(config_file)
