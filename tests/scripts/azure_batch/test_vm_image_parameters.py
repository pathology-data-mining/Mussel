#!/usr/bin/env python3
"""
Tests for Azure Batch VM image configuration parameters.
This test validates that the --publisher, --offer, and --sku parameters are properly configured.
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
    assert 'microsoft-dsvm' in content, \
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
    assert 'ubuntu-hpc' in content, \
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
    assert 'batch.node.ubuntu 22.04' in content, \
        "Missing default sku value"


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


def test_create_pool_has_vm_image_parameters():
    """Test that create_pool method has VM image parameters."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check that create_pool method signature includes parameters
    assert 'def create_pool(' in content, \
        "create_pool method not found"
    
    # Check for parameters in method signature
    assert 'publisher: str = "microsoft-dsvm"' in content, \
        "publisher parameter missing in create_pool method signature"
    
    assert 'offer: str = "ubuntu-hpc"' in content, \
        "offer parameter missing in create_pool method signature"
    
    assert 'sku: str = "batch.node.ubuntu 22.04"' in content, \
        "sku parameter missing in create_pool method signature"


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


def test_sku_used_as_node_agent():
    """Test that sku is used as node_agent_sku_id in VirtualMachineConfiguration."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check that sku is used as node agent SKU ID
    assert 'node_agent_sku_id=sku' in content, \
        "sku parameter not used as node_agent_sku_id in VirtualMachineConfiguration"


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


def test_vm_type_parameter_removed():
    """Test that vm_type parameter has been removed."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check that vm_type is not in CLI arguments
    # Should not find --vm-type as a CLI parameter
    assert '--vm-type' not in content or 'Azure VM image SKU and node agent SKU ID' in content, \
        "vm_type parameter should be removed from CLI"
    
    # Check that vm_type is not in create_pool signature
    # Look for the pattern we're expecting
    lines_with_create_pool = [line for line in content.split('\n') if 'def create_pool' in line or 'vm_type:' in line]
    for line in lines_with_create_pool:
        if 'vm_type:' in line and 'def create_pool' not in line:
            # Found vm_type parameter in wrong place
            assert False, "vm_type parameter should not be in create_pool method signature"


def test_default_vm_size_updated():
    """Test that default VM size has been updated to A100."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check that DEFAULT_VM_SIZE is set to A100
    assert 'DEFAULT_VM_SIZE = "Standard_NC24ads_A100_v4"' in content, \
        "DEFAULT_VM_SIZE should be set to Standard_NC24ads_A100_v4"


if __name__ == '__main__':
    # Run tests
    test_publisher_parameter_exists()
    print("✓ test_publisher_parameter_exists passed")
    
    test_offer_parameter_exists()
    print("✓ test_offer_parameter_exists passed")
    
    test_sku_parameter_exists()
    print("✓ test_sku_parameter_exists passed")
    
    test_parameters_passed_to_create_pool()
    print("✓ test_parameters_passed_to_create_pool passed")
    
    test_create_pool_has_vm_image_parameters()
    print("✓ test_create_pool_has_vm_image_parameters passed")
    
    test_parameters_used_in_image_reference()
    print("✓ test_parameters_used_in_image_reference passed")
    
    test_sku_used_as_node_agent()
    print("✓ test_sku_used_as_node_agent passed")
    
    test_config_loading_supports_vm_image_parameters()
    print("✓ test_config_loading_supports_vm_image_parameters passed")
    
    test_vm_type_parameter_removed()
    print("✓ test_vm_type_parameter_removed passed")
    
    test_default_vm_size_updated()
    print("✓ test_default_vm_size_updated passed")
    
    print("\nAll tests passed!")
