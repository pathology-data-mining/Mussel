#!/usr/bin/env python3
"""
Tests for Azure Batch configuration file parameter loading.
This test validates that Azure-specific parameters are properly read from config files.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Get absolute paths using pathlib
TEST_DIR = Path(__file__).parent
SCRIPTS_DIR = TEST_DIR / '..' / '..' / '..' / 'scripts'
COMMON_DIR = SCRIPTS_DIR / 'common'

# Add scripts directory to path for imports
sys.path.insert(0, str(COMMON_DIR))

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from config_loader import load_config_defaults


def test_azure_section_loaded_from_yaml():
    """Test that azure section parameters are loaded from YAML config."""
    if not HAS_YAML:
        print("SKIP: test_azure_section_loaded_from_yaml (PyYAML not installed)")
        return
    
    config_content = """
prefilter_model_type: CTRANSPATH
batch_size: 64

azure:
  container_image: "custom/image:tag"
  storage_account_name: "teststorage"
  vm_size: "Standard_NC12s_v3"
  node_count: 4
  enable_auto_scale: true
  min_node_count: 2
  max_node_count: 10
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        # Load config with backend='azure'
        params = load_config_defaults(config_file, backend='azure')
        
        # Verify azure-specific parameters were merged
        assert 'container_image' in params, "container_image not found in loaded params"
        assert params['container_image'] == "custom/image:tag", \
            f"container_image mismatch: {params['container_image']}"
        
        assert 'storage_account_name' in params, "storage_account_name not found"
        assert params['storage_account_name'] == "teststorage", \
            f"storage_account_name mismatch: {params['storage_account_name']}"
        
        assert 'vm_size' in params, "vm_size not found"
        assert params['vm_size'] == "Standard_NC12s_v3", \
            f"vm_size mismatch: {params['vm_size']}"
        
        assert 'node_count' in params, "node_count not found"
        assert params['node_count'] == 4, \
            f"node_count mismatch: {params['node_count']}"
        
        assert 'enable_auto_scale' in params, "enable_auto_scale not found"
        assert params['enable_auto_scale'] == True, \
            f"enable_auto_scale mismatch: {params['enable_auto_scale']}"
        
        assert 'min_node_count' in params, "min_node_count not found"
        assert params['min_node_count'] == 2, \
            f"min_node_count mismatch: {params['min_node_count']}"
        
        assert 'max_node_count' in params, "max_node_count not found"
        assert params['max_node_count'] == 10, \
            f"max_node_count mismatch: {params['max_node_count']}"
        
        # Also verify general parameters are present
        assert params['prefilter_model_type'] == "CTRANSPATH"
        assert params['batch_size'] == 64
        
    finally:
        os.unlink(config_file)


def test_azure_section_loaded_from_json():
    """Test that azure section parameters are loaded from JSON config."""
    config_content = {
        "prefilter_model_type": "CTRANSPATH",
        "batch_size": 32,
        "azure": {
            "container_image": "another/image:v2",
            "storage_account_name": "jsonstorage",
            "vm_size": "Standard_NC24s_v3"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_content, f)
        config_file = f.name
    
    try:
        # Load config with backend='azure'
        params = load_config_defaults(config_file, backend='azure')
        
        # Verify azure-specific parameters were merged
        assert 'container_image' in params, "container_image not found in loaded params"
        assert params['container_image'] == "another/image:v2", \
            f"container_image mismatch: {params['container_image']}"
        
        assert 'storage_account_name' in params, "storage_account_name not found"
        assert params['storage_account_name'] == "jsonstorage", \
            f"storage_account_name mismatch: {params['storage_account_name']}"
        
        assert 'vm_size' in params, "vm_size not found"
        assert params['vm_size'] == "Standard_NC24s_v3", \
            f"vm_size mismatch: {params['vm_size']}"
        
    finally:
        os.unlink(config_file)


def test_azure_batch_alias():
    """Test that azure_batch section works as an alias for azure."""
    if not HAS_YAML:
        print("SKIP: test_azure_batch_alias (PyYAML not installed)")
        return
    
    config_content = """
prefilter_model_type: VIRCHOW

azure_batch:
  container_image: "alias/test:latest"
  vm_size: "Standard_NC6s_v3"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        # Load config with backend='azure'
        params = load_config_defaults(config_file, backend='azure')
        
        # Verify azure_batch parameters were merged
        assert params['container_image'] == "alias/test:latest", \
            f"container_image from azure_batch not loaded: {params.get('container_image')}"
        
        assert params['vm_size'] == "Standard_NC6s_v3", \
            f"vm_size from azure_batch not loaded: {params.get('vm_size')}"
        
    finally:
        os.unlink(config_file)


def test_azure_params_without_backend():
    """Test that azure section is NOT loaded when backend is not specified."""
    if not HAS_YAML:
        print("SKIP: test_azure_params_without_backend (PyYAML not installed)")
        return
    
    config_content = """
prefilter_model_type: CTRANSPATH
batch_size: 64

azure:
  container_image: "custom/image:tag"
  storage_account_name: "teststorage"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        # Load config WITHOUT backend parameter
        params = load_config_defaults(config_file)
        
        # Azure-specific parameters should NOT be in top-level params
        assert 'container_image' not in params, \
            "container_image should not be loaded without backend='azure'"
        
        assert 'storage_account_name' not in params, \
            "storage_account_name should not be loaded without backend='azure'"
        
        # General parameters should still be present
        assert params['prefilter_model_type'] == "CTRANSPATH"
        assert params['batch_size'] == 64
        
    finally:
        os.unlink(config_file)


def test_azure_params_override_general():
    """Test that azure section parameters override general parameters."""
    if not HAS_YAML:
        print("SKIP: test_azure_params_override_general (PyYAML not installed)")
        return
    
    config_content = """
prefilter_model_type: CTRANSPATH
container_image: "general/image:default"
node_count: 1

azure:
  container_image: "azure/image:specific"
  node_count: 8
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        # Load config with backend='azure'
        params = load_config_defaults(config_file, backend='azure')
        
        # Azure-specific parameters should override general ones
        assert params['container_image'] == "azure/image:specific", \
            f"Azure container_image should override general: {params['container_image']}"
        
        assert params['node_count'] == 8, \
            f"Azure node_count should override general: {params['node_count']}"
        
    finally:
        os.unlink(config_file)


if __name__ == '__main__':
    # Run tests
    test_azure_section_loaded_from_yaml()
    print("✓ test_azure_section_loaded_from_yaml passed")
    
    test_azure_section_loaded_from_json()
    print("✓ test_azure_section_loaded_from_json passed")
    
    test_azure_batch_alias()
    print("✓ test_azure_batch_alias passed")
    
    test_azure_params_without_backend()
    print("✓ test_azure_params_without_backend passed")
    
    test_azure_params_override_general()
    print("✓ test_azure_params_override_general passed")
    
    print("\nAll tests passed!")
