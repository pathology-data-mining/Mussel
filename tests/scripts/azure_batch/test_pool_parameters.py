#!/usr/bin/env python3
"""
Tests for Azure Batch pool creation parameters.
This test validates that the --use-gpu and --no-gpu parameters are properly configured.
"""

import sys
import re
from pathlib import Path

# Get absolute paths using pathlib
TEST_DIR = Path(__file__).parent
SCRIPTS_DIR = TEST_DIR / '..' / '..' / '..' / 'scripts' / 'azure_batch'


def test_use_gpu_parameter_exists():
    """Test that --use-gpu parameter is defined in the CLI."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check for --use-gpu argument
    assert '--use-gpu' in content, \
        "Missing --use-gpu CLI parameter"
    
    # Check for use-gpu with appropriate configuration (more flexible matching)
    assert 'store_true' in content and 'Enable GPU support' in content, \
        "Missing or incorrect --use-gpu parameter configuration"


def test_no_gpu_parameter_exists():
    """Test that --no-gpu parameter is defined in the CLI."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check for --no-gpu argument
    assert '--no-gpu' in content, \
        "Missing --no-gpu CLI parameter"
    
    # Check that it sets use_gpu to False (more flexible matching)
    assert 'use_gpu' in content and 'store_false' in content and 'Disable GPU support' in content, \
        "Missing or incorrect --no-gpu parameter configuration"


def test_use_gpu_passed_to_create_pool():
    """Test that use_gpu parameter is passed to create_pool method."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Look for use_gpu being passed to create_pool (flexible pattern)
    # This checks that somewhere in the file, create_pool is called with use_gpu=args.use_gpu
    assert 'create_pool(' in content and 'use_gpu=args.use_gpu' in content, \
        "use_gpu parameter not passed to create_pool method"


def test_create_pool_has_use_gpu_parameter():
    """Test that create_pool method has use_gpu parameter."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check that create_pool method signature includes use_gpu
    assert 'def create_pool(' in content, \
        "create_pool method not found"
    
    # Check for use_gpu parameter in method signature
    assert 'use_gpu: bool = True' in content, \
        "use_gpu parameter missing in create_pool method signature"


def test_delete_pool_help_text_updated():
    """Test that --delete-pool help text clarifies behavior with --monitor."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check for improved help text
    assert 'Delete pool after completion' in content, \
        "Missing basic delete-pool help text"
    
    # Check that it mentions monitoring behavior
    assert 'When used with --monitor' in content, \
        "delete-pool help text doesn't mention --monitor behavior"


def test_gpu_vm_prefixes_constant_exists():
    """Test that GPU_VM_PREFIXES constant is defined in the class."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check for GPU_VM_PREFIXES constant
    assert 'GPU_VM_PREFIXES' in content, \
        "Missing GPU_VM_PREFIXES constant"
    
    # Check that it contains expected GPU VM families
    assert 'Standard_NC' in content and 'Standard_ND' in content and 'Standard_NV' in content, \
        "GPU_VM_PREFIXES missing expected VM family prefixes"


if __name__ == '__main__':
    # Run tests
    test_use_gpu_parameter_exists()
    print("✓ test_use_gpu_parameter_exists passed")
    
    test_no_gpu_parameter_exists()
    print("✓ test_no_gpu_parameter_exists passed")
    
    test_use_gpu_passed_to_create_pool()
    print("✓ test_use_gpu_passed_to_create_pool passed")
    
    test_create_pool_has_use_gpu_parameter()
    print("✓ test_create_pool_has_use_gpu_parameter passed")
    
    test_delete_pool_help_text_updated()
    print("✓ test_delete_pool_help_text_updated passed")
    
    test_gpu_vm_prefixes_constant_exists()
    print("✓ test_gpu_vm_prefixes_constant_exists passed")
    
    print("\nAll tests passed!")
