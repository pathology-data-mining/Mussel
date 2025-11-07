#!/usr/bin/env python3
"""
Test seg_config grouping in YAML configuration files.
"""

import tempfile
import os
from pathlib import Path
import sys

# Add scripts/common to path
SCRIPTS_COMMON = Path(__file__).parent.parent.parent / 'scripts' / 'common'
sys.path.insert(0, str(SCRIPTS_COMMON))

from config_loader import load_config_defaults


def test_seg_config_group_in_yaml():
    """Test that seg_config.group is properly loaded and renamed to seg_config_group."""
    yaml_content = """
seg_config:
  group: biopsy

prefilter_model_type: CTRANSPATH
batch_size: 64
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_file = f.name
    
    try:
        params = load_config_defaults(config_file)
        
        assert 'seg_config_group' in params, \
            "seg_config_group not found in loaded parameters"
        assert params['seg_config_group'] == 'biopsy', \
            f"Expected seg_config_group='biopsy', got '{params['seg_config_group']}'"
        
        print("✓ test_seg_config_group_in_yaml passed")
    finally:
        os.unlink(config_file)


def test_seg_config_individual_params_in_yaml():
    """Test that individual seg_config parameters are properly flattened."""
    yaml_content = """
seg_config:
  patch_size: 512
  mpp: 0.25
  segment_threshold: 15
  median_blur_ksize: 11
  use_otsu: true

prefilter_model_type: CTRANSPATH
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_file = f.name
    
    try:
        params = load_config_defaults(config_file)
        
        assert 'patch_size' in params, "patch_size not found in loaded parameters"
        assert params['patch_size'] == 512, f"Expected patch_size=512, got {params['patch_size']}"
        
        assert 'mpp' in params, "mpp not found in loaded parameters"
        assert params['mpp'] == 0.25, f"Expected mpp=0.25, got {params['mpp']}"
        
        assert 'segment_threshold' in params, "segment_threshold not found in loaded parameters"
        assert params['segment_threshold'] == 15, f"Expected segment_threshold=15, got {params['segment_threshold']}"
        
        assert 'median_blur_ksize' in params, "median_blur_ksize not found in loaded parameters"
        assert params['median_blur_ksize'] == 11, f"Expected median_blur_ksize=11, got {params['median_blur_ksize']}"
        
        assert 'use_otsu' in params, "use_otsu not found in loaded parameters"
        assert params['use_otsu'] is True, f"Expected use_otsu=True, got {params['use_otsu']}"
        
        print("✓ test_seg_config_individual_params_in_yaml passed")
    finally:
        os.unlink(config_file)


def test_seg_config_group_and_individual_params():
    """Test that both group and individual params can be specified (individual params available)."""
    yaml_content = """
seg_config:
  group: tcga
  patch_size: 384
  mpp: 0.75

prefilter_model_type: CTRANSPATH
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_file = f.name
    
    try:
        params = load_config_defaults(config_file)
        
        assert 'seg_config_group' in params, "seg_config_group not found in loaded parameters"
        assert params['seg_config_group'] == 'tcga', f"Expected seg_config_group='tcga', got '{params['seg_config_group']}'"
        
        # Individual params should also be present (they can override group defaults)
        assert 'patch_size' in params, "patch_size not found in loaded parameters"
        assert params['patch_size'] == 384, f"Expected patch_size=384, got {params['patch_size']}"
        
        assert 'mpp' in params, "mpp not found in loaded parameters"
        assert params['mpp'] == 0.75, f"Expected mpp=0.75, got {params['mpp']}"
        
        print("✓ test_seg_config_group_and_individual_params passed")
    finally:
        os.unlink(config_file)


def test_backward_compatibility_flat_params():
    """Test that flat parameters (old style) still work."""
    yaml_content = """
patch_size: 256
mpp: 0.5
segment_threshold: 0

prefilter_model_type: CTRANSPATH
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_file = f.name
    
    try:
        params = load_config_defaults(config_file)
        
        # Flat params should still be loaded
        assert 'patch_size' in params, "patch_size not found in loaded parameters"
        assert params['patch_size'] == 256, f"Expected patch_size=256, got {params['patch_size']}"
        
        assert 'mpp' in params, "mpp not found in loaded parameters"
        assert params['mpp'] == 0.5, f"Expected mpp=0.5, got {params['mpp']}"
        
        assert 'segment_threshold' in params, "segment_threshold not found in loaded parameters"
        assert params['segment_threshold'] == 0, f"Expected segment_threshold=0, got {params['segment_threshold']}"
        
        print("✓ test_backward_compatibility_flat_params passed")
    finally:
        os.unlink(config_file)


def test_seg_config_all_parameters():
    """Test that all seg_config parameters are correctly flattened."""
    yaml_content = """
seg_config:
  segment_threshold: 20
  patch_size: 256
  step_size: 128
  mpp: 0.5
  seg_level: -1
  segment_max_value: 255
  median_blur_ksize: 7
  morphology_ex_kernel: 2
  ref_patch_size: 512
  use_otsu: false
  tissue_area_threshold: 100
  hole_area_threshold: 16
  max_num_holes: 8
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_file = f.name
    
    try:
        params = load_config_defaults(config_file)
        
        expected_params = {
            'segment_threshold': 20,
            'patch_size': 256,
            'step_size': 128,
            'mpp': 0.5,
            'seg_level': -1,
            'segment_max_value': 255,
            'median_blur_ksize': 7,
            'morphology_ex_kernel': 2,
            'ref_patch_size': 512,
            'use_otsu': False,
            'tissue_area_threshold': 100,
            'hole_area_threshold': 16,
            'max_num_holes': 8,
        }
        
        for param_name, expected_value in expected_params.items():
            assert param_name in params, f"{param_name} not found in loaded parameters"
            assert params[param_name] == expected_value, \
                f"Expected {param_name}={expected_value}, got {params[param_name]}"
        
        print("✓ test_seg_config_all_parameters passed")
    finally:
        os.unlink(config_file)


if __name__ == '__main__':
    try:
        import yaml
    except ImportError:
        print("WARNING: PyYAML not installed, skipping YAML tests")
        print("Install with: pip install PyYAML")
        sys.exit(0)
    
    tests = [
        test_seg_config_group_in_yaml,
        test_seg_config_individual_params_in_yaml,
        test_seg_config_group_and_individual_params,
        test_backward_compatibility_flat_params,
        test_seg_config_all_parameters,
    ]
    
    failed = []
    for test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed.append((test_func.__name__, str(e)))
        except Exception as e:
            print(f"✗ {test_func.__name__} error: {e}")
            failed.append((test_func.__name__, str(e)))
    
    if failed:
        print(f"\n{len(failed)} test(s) failed:")
        for test_name, error in failed:
            print(f"  - {test_name}: {error}")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests passed!")
