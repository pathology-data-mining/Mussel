#!/usr/bin/env python3
"""
Tests for Azure Batch VM image configuration parameters.
This test validates that the --publisher, --offer, --sku, and --vm-type parameters are properly configured.
"""

import sys
import re
from pathlib import Path

# Get absolute paths using pathlib
TEST_DIR = Path(__file__).parent
SCRIPTS_DIR = TEST_DIR / '..' / '..' / '..' / 'scripts' / 'azure_batch'


def test_publisher_parameter_exists():
    """Test that --publisher parameter is defined in the CLI."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check for --publisher argument
    assert '--publisher' in content, \
        "Missing --publisher CLI parameter"
    
    # Check for publisher with default configuration
    assert 'microsoft-azure-batch' in content, \
        "Missing default publisher value"


def test_offer_parameter_exists():
    """Test that --offer parameter is defined in the CLI."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check for --offer argument
    assert '--offer' in content, \
        "Missing --offer CLI parameter"
    
    # Check for offer with default configuration
    assert 'ubuntu-server-container' in content, \
        "Missing default offer value"


def test_sku_parameter_exists():
    """Test that --sku parameter is defined in the CLI."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check for --sku argument
    assert '--sku' in content, \
        "Missing --sku CLI parameter"
    
    # Check for sku with default configuration
    assert '20-04-lts' in content, \
        "Missing default sku value"


def test_vm_type_parameter_exists():
    """Test that --vm-type parameter is defined in the CLI."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check for --vm-type argument
    assert '--vm-type' in content, \
        "Missing --vm-type CLI parameter"
    
    # Check for vm-type with default configuration
    assert 'batch.node.ubuntu 20.04' in content, \
        "Missing default vm_type value"


def test_parameters_passed_to_create_pool():
    """Test that VM image parameters are passed to create_pool method."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Look for parameters being passed to create_pool
    assert 'publisher=args.publisher' in content, \
        "publisher parameter not passed to create_pool method"
    
    assert 'offer=args.offer' in content, \
        "offer parameter not passed to create_pool method"
    
    assert 'sku=args.sku' in content, \
        "sku parameter not passed to create_pool method"
    
    assert 'vm_type=args.vm_type' in content, \
        "vm_type parameter not passed to create_pool method"


def test_create_pool_has_vm_image_parameters():
    """Test that create_pool method has VM image parameters."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check that create_pool method signature includes parameters
    assert 'def create_pool(' in content, \
        "create_pool method not found"
    
    # Check for parameters in method signature
    assert 'publisher: str = "microsoft-azure-batch"' in content, \
        "publisher parameter missing in create_pool method signature"
    
    assert 'offer: str = "ubuntu-server-container"' in content, \
        "offer parameter missing in create_pool method signature"
    
    assert 'sku: str = "20-04-lts"' in content, \
        "sku parameter missing in create_pool method signature"
    
    assert 'vm_type: str = "batch.node.ubuntu 20.04"' in content, \
        "vm_type parameter missing in create_pool method signature"


def test_parameters_used_in_image_reference():
    """Test that parameters are used in ImageReference creation."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check that parameters are used (not hardcoded) in ImageReference
    # Look for pattern like: publisher=publisher, offer=offer, sku=sku
    assert 'publisher=publisher' in content, \
        "publisher parameter not used in ImageReference"
    
    assert 'offer=offer' in content, \
        "offer parameter not used in ImageReference"
    
    assert 'sku=sku' in content, \
        "sku parameter not used in ImageReference"


def test_vm_type_used_in_vm_config():
    """Test that vm_type is used in VirtualMachineConfiguration."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check that vm_type is used (not hardcoded)
    assert 'node_agent_sku_id=vm_type' in content, \
        "vm_type parameter not used in VirtualMachineConfiguration"


def test_config_loading_supports_vm_image_parameters():
    """Test that config file loading supports VM image parameters."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check that config defaults are loaded for VM image parameters
    assert "'publisher' in config_defaults" in content, \
        "Config loading doesn't support publisher parameter"
    
    assert "'offer' in config_defaults" in content, \
        "Config loading doesn't support offer parameter"
    
    assert "'sku' in config_defaults" in content, \
        "Config loading doesn't support sku parameter"
    
    assert "'vm_type' in config_defaults" in content, \
        "Config loading doesn't support vm_type parameter"


if __name__ == '__main__':
    # Run tests
    test_publisher_parameter_exists()
    print("✓ test_publisher_parameter_exists passed")
    
    test_offer_parameter_exists()
    print("✓ test_offer_parameter_exists passed")
    
    test_sku_parameter_exists()
    print("✓ test_sku_parameter_exists passed")
    
    test_vm_type_parameter_exists()
    print("✓ test_vm_type_parameter_exists passed")
    
    test_parameters_passed_to_create_pool()
    print("✓ test_parameters_passed_to_create_pool passed")
    
    test_create_pool_has_vm_image_parameters()
    print("✓ test_create_pool_has_vm_image_parameters passed")
    
    test_parameters_used_in_image_reference()
    print("✓ test_parameters_used_in_image_reference passed")
    
    test_vm_type_used_in_vm_config()
    print("✓ test_vm_type_used_in_vm_config passed")
    
    test_config_loading_supports_vm_image_parameters()
    print("✓ test_config_loading_supports_vm_image_parameters passed")
    
    print("\nAll tests passed!")
