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
    
    # Check for use-gpu action and default value
    assert 'action="store_true", default=True' in content and \
           'Enable GPU support for pool nodes (default: True)' in content, \
        "Missing or incorrect --use-gpu parameter configuration"


def test_no_gpu_parameter_exists():
    """Test that --no-gpu parameter is defined in the CLI."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check for --no-gpu argument
    assert '--no-gpu' in content, \
        "Missing --no-gpu CLI parameter"
    
    # Check that it sets use_gpu to False
    assert 'dest="use_gpu", action="store_false"' in content and \
           'Disable GPU support for pool nodes' in content, \
        "Missing or incorrect --no-gpu parameter configuration"


def test_use_gpu_passed_to_create_pool():
    """Test that use_gpu parameter is passed to create_pool method."""
    submit_file = SCRIPTS_DIR / 'submit_batch_jobs.py'
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Find the create_pool call in main()
    # Look for the pattern where create_pool is called with use_gpu
    pattern = r'submitter\.create_pool\([^)]*use_gpu=args\.use_gpu'
    
    assert re.search(pattern, content, re.DOTALL), \
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
    
    print("\nAll tests passed!")
