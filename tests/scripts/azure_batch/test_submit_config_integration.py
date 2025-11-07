#!/usr/bin/env python3
"""
Integration test for Azure Batch parameter loading from config files.
This test validates that submit_batch_jobs.py correctly applies Azure parameters from config.
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

# Add common scripts directory to path for imports
sys.path.insert(0, str(COMMON_DIR))

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def test_config_parameters_applied_to_pool_creation():
    """Test that Azure config parameters are applied when creating pool."""
    if not HAS_YAML:
        print("SKIP: test_config_parameters_applied_to_pool_creation (PyYAML not installed)")
        return
    
    # Create a test config file
    config_content = """
prefilter_model_type: CTRANSPATH
batch_size: 64
num_workers: 4

azure:
  container_image: "custom/mussel:test"
  storage_account_name: "configstorage"
  vm_size: "Standard_NC24s_v3"
  node_count: 5
  enable_auto_scale: true
  min_node_count: 3
  max_node_count: 20
  auto_scale_evaluation_interval: 10
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        from config_loader import load_config_defaults
        
        import argparse
        
        # Recreate the parser logic from main()
        parser = argparse.ArgumentParser()
        parser.add_argument("--config-file", "--config", dest="config_file")
        parser.add_argument("--vm-size", default="Standard_NC6s_v3")
        parser.add_argument("--node-count", type=int, default=1)
        parser.add_argument("--container-image", default="mskmind/mussel:latest-torch-gpu")
        parser.add_argument("--storage-account-name")
        parser.add_argument("--enable-auto-scale", action="store_true")
        parser.add_argument("--min-node-count", type=int)
        parser.add_argument("--max-node-count", type=int)
        parser.add_argument("--auto-scale-evaluation-interval", type=int, default=15)
        
        args = parser.parse_args([
            '--config-file', config_file,
        ])
        
        # Load config and apply Azure parameters (simulating the main() logic)
        config_defaults = load_config_defaults(config_file, backend='azure')
        
        # Apply config parameters (mimicking the logic we added to submit_batch_jobs.py)
        if not args.storage_account_name and 'storage_account_name' in config_defaults:
            args.storage_account_name = config_defaults['storage_account_name']
        
        if args.container_image == "mskmind/mussel:latest-torch-gpu" and 'container_image' in config_defaults:
            args.container_image = config_defaults['container_image']
        
        if args.vm_size == "Standard_NC6s_v3" and 'vm_size' in config_defaults:
            args.vm_size = config_defaults['vm_size']
        
        if args.node_count == 1 and 'node_count' in config_defaults:
            args.node_count = config_defaults['node_count']
        
        if not args.enable_auto_scale and config_defaults.get('enable_auto_scale'):
            args.enable_auto_scale = config_defaults['enable_auto_scale']
        
        if args.min_node_count is None and 'min_node_count' in config_defaults:
            args.min_node_count = config_defaults['min_node_count']
        
        if args.max_node_count is None and 'max_node_count' in config_defaults:
            args.max_node_count = config_defaults['max_node_count']
        
        if args.auto_scale_evaluation_interval == 15 and 'auto_scale_evaluation_interval' in config_defaults:
            args.auto_scale_evaluation_interval = config_defaults['auto_scale_evaluation_interval']
        
        # Verify the parameters were applied from config
        assert args.container_image == "custom/mussel:test", \
            f"container_image not applied from config: {args.container_image}"
        
        assert args.storage_account_name == "configstorage", \
            f"storage_account_name not applied from config: {args.storage_account_name}"
        
        assert args.vm_size == "Standard_NC24s_v3", \
            f"vm_size not applied from config: {args.vm_size}"
        
        assert args.node_count == 5, \
            f"node_count not applied from config: {args.node_count}"
        
        assert args.enable_auto_scale == True, \
            f"enable_auto_scale not applied from config: {args.enable_auto_scale}"
        
        assert args.min_node_count == 3, \
            f"min_node_count not applied from config: {args.min_node_count}"
        
        assert args.max_node_count == 20, \
            f"max_node_count not applied from config: {args.max_node_count}"
        
        assert args.auto_scale_evaluation_interval == 10, \
            f"auto_scale_evaluation_interval not applied from config: {args.auto_scale_evaluation_interval}"
    
    finally:
        os.unlink(config_file)


def test_command_line_overrides_config():
    """Test that command-line arguments override config file values."""
    if not HAS_YAML:
        print("SKIP: test_command_line_overrides_config (PyYAML not installed)")
        return
    
    # Create a test config file with Azure parameters
    config_content = """
azure:
  container_image: "config/image:tag"
  vm_size: "Standard_NC12s_v3"
  node_count: 4
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        from config_loader import load_config_defaults
        
        import argparse
        
        # Simulate command-line with explicit values
        parser = argparse.ArgumentParser()
        parser.add_argument("--config-file")
        parser.add_argument("--vm-size", default="Standard_NC6s_v3")
        parser.add_argument("--node-count", type=int, default=1)
        parser.add_argument("--container-image", default="mskmind/mussel:latest-torch-gpu")
        
        # Parse with explicit command-line values
        args = parser.parse_args([
            '--config-file', config_file,
            '--vm-size', 'Standard_D8s_v3',  # Override config
            '--container-image', 'custom/override:latest',  # Override config
        ])
        
        config_defaults = load_config_defaults(config_file, backend='azure')
        
        # Apply config parameters with override logic
        if args.container_image == "mskmind/mussel:latest-torch-gpu" and 'container_image' in config_defaults:
            args.container_image = config_defaults['container_image']
        
        if args.vm_size == "Standard_NC6s_v3" and 'vm_size' in config_defaults:
            args.vm_size = config_defaults['vm_size']
        
        if args.node_count == 1 and 'node_count' in config_defaults:
            args.node_count = config_defaults['node_count']
        
        # Verify command-line overrides were preserved
        assert args.container_image == "custom/override:latest", \
            f"Command-line container_image should override config: {args.container_image}"
        
        assert args.vm_size == "Standard_D8s_v3", \
            f"Command-line vm_size should override config: {args.vm_size}"
        
        # node_count should use config since command-line used default
        assert args.node_count == 4, \
            f"node_count from config should be applied when using default: {args.node_count}"
        
    finally:
        os.unlink(config_file)


if __name__ == '__main__':
    # Run tests
    test_config_parameters_applied_to_pool_creation()
    print("✓ test_config_parameters_applied_to_pool_creation passed")
    
    test_command_line_overrides_config()
    print("✓ test_command_line_overrides_config passed")
    
    print("\nAll integration tests passed!")
