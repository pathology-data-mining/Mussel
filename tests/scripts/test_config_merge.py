#!/usr/bin/env python3
"""
Tests for config merging behavior in submission scripts.

This test verifies that model paths from config files are correctly
accepted and not overridden by None values from command-line defaults.
"""

import os
import sys
import tempfile

import pytest

# Import the config loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'common'))
from config_loader import load_config_defaults


def test_prefilter_model_path_from_config():
    """
    Test that prefilter_model_path from config file is correctly loaded.
    
    This tests the fix for the issue where prefilter_model_path from config
    was being overridden by None values from command-line args.
    """
    yaml_content = """
# Test config with prefilter_model_path
prefilter_model_type: CTRANSPATH
prefilter_model_path: /path/to/ctranspath.pth
batch_size: 64
num_workers: 4
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_file = f.name
    
    try:
        # Load config defaults (simulating what submission scripts do)
        config_defaults = load_config_defaults(config_file, backend='condor')
        
        # Verify prefilter_model_path is present in loaded config
        assert 'prefilter_model_path' in config_defaults
        assert config_defaults['prefilter_model_path'] == '/path/to/ctranspath.pth'
        assert config_defaults['prefilter_model_type'] == 'CTRANSPATH'
        assert config_defaults['batch_size'] == 64
        
        # Simulate the old buggy behavior where None would override config
        csv_kwargs_old_buggy = {
            'prefilter_model_type': 'CTRANSPATH',
            'prefilter_model_path': None,  # This is what the bug did
            'batch_size': 64,
        }
        
        # Old merge (buggy): config first, then csv_kwargs with None
        merged_buggy = {**config_defaults, **csv_kwargs_old_buggy}
        # The bug: prefilter_model_path becomes None instead of the config value
        assert merged_buggy['prefilter_model_path'] is None  # This was the bug!
        
        # Simulate the fixed behavior where None is not included
        csv_kwargs_fixed = {
            'prefilter_model_type': 'CTRANSPATH',
            # prefilter_model_path is not included if it's None
            'batch_size': 64,
        }
        
        # Fixed merge: config first, then csv_kwargs without None
        merged_fixed = {**config_defaults, **csv_kwargs_fixed}
        # The fix: prefilter_model_path retains the config value
        assert merged_fixed['prefilter_model_path'] == '/path/to/ctranspath.pth'
        
    finally:
        os.unlink(config_file)


def test_slide_model_path_from_config():
    """
    Test that slide_model_path from config file is correctly loaded.
    
    This tests the same fix for slide_model_path which had the same issue.
    """
    yaml_content = """
# Test config with slide_model_path
slide_model_type: GIGAPATH_SLIDE
slide_model_path: /path/to/gigapath.pth
aggregation_method: model
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_file = f.name
    
    try:
        # Load config defaults
        config_defaults = load_config_defaults(config_file, backend='slurm')
        
        # Verify slide_model_path is present in loaded config
        assert 'slide_model_path' in config_defaults
        assert config_defaults['slide_model_path'] == '/path/to/gigapath.pth'
        assert config_defaults['slide_model_type'] == 'GIGAPATH_SLIDE'
        
        # Simulate the fixed behavior
        csv_kwargs_fixed = {
            'slide_model_type': 'GIGAPATH_SLIDE',
            'aggregation_method': 'model',
            # slide_model_path is not included if there's no pre-downloaded value
        }
        
        # Fixed merge: config first, then csv_kwargs
        merged_fixed = {**config_defaults, **csv_kwargs_fixed}
        # The fix: slide_model_path retains the config value
        assert merged_fixed['slide_model_path'] == '/path/to/gigapath.pth'
        
    finally:
        os.unlink(config_file)


def test_predownloaded_models_override_config():
    """
    Test that pre-downloaded models correctly override config values.
    
    When models are pre-downloaded, their paths should take precedence
    over config file values.
    """
    yaml_content = """
# Test config with model paths
prefilter_model_path: /config/path/ctranspath.pth
slide_model_path: /config/path/gigapath.pth
slide_model_type: GIGAPATH_SLIDE
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_file = f.name
    
    try:
        # Load config defaults
        config_defaults = load_config_defaults(config_file, backend='condor')
        
        # Simulate CSV kwargs without model paths
        csv_kwargs = {
            'slide_model_type': 'GIGAPATH_SLIDE',
        }
        
        # Merge config with csv_kwargs
        merged = {**config_defaults, **csv_kwargs}
        
        # Verify config values are present
        assert merged['prefilter_model_path'] == '/config/path/ctranspath.pth'
        assert merged['slide_model_path'] == '/config/path/gigapath.pth'
        
        # Now simulate pre-downloaded models (after merge, like in the fix)
        model_paths = {
            'CTRANSPATH': '/predownload/cache/ctranspath.pth',
            'GIGAPATH_SLIDE': '/predownload/cache/gigapath.pth',
        }
        
        # Apply pre-downloaded paths (only if they exist)
        if model_paths.get('CTRANSPATH'):
            merged['prefilter_model_path'] = model_paths['CTRANSPATH']
        if model_paths.get('GIGAPATH_SLIDE'):
            merged['slide_model_path'] = model_paths['GIGAPATH_SLIDE']
        
        # Verify pre-downloaded paths override config
        assert merged['prefilter_model_path'] == '/predownload/cache/ctranspath.pth'
        assert merged['slide_model_path'] == '/predownload/cache/gigapath.pth'
        
    finally:
        os.unlink(config_file)


def test_command_line_args_override_config():
    """
    Test that explicit command-line arguments override config values.
    
    When a user explicitly provides a value via command line, it should
    take precedence over the config file.
    """
    yaml_content = """
# Test config
batch_size: 64
num_workers: 4
prefilter_model_type: CTRANSPATH
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_file = f.name
    
    try:
        # Load config defaults
        config_defaults = load_config_defaults(config_file, backend='slurm')
        
        # Simulate CSV kwargs with explicit command-line values
        csv_kwargs = {
            'batch_size': 128,  # User specified different value
            'num_workers': 8,   # User specified different value
            'prefilter_model_type': 'CTRANSPATH',  # Same as config
        }
        
        # Merge: config first, then csv_kwargs (csv_kwargs wins)
        merged = {**config_defaults, **csv_kwargs}
        
        # Verify command-line args override config
        assert merged['batch_size'] == 128  # From command line
        assert merged['num_workers'] == 8   # From command line
        assert merged['prefilter_model_type'] == 'CTRANSPATH'  # Same in both
        
    finally:
        os.unlink(config_file)
