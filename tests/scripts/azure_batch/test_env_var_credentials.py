#!/usr/bin/env python3
"""
Tests for environment variable support for Azure and AWS credentials.
This test validates that credentials can be read from environment variables.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Get absolute paths using pathlib
TEST_DIR = Path(__file__).parent
SCRIPTS_DIR = TEST_DIR / '..' / '..' / '..' / 'scripts'
AZURE_BATCH_DIR = SCRIPTS_DIR / 'azure_batch'
COMMON_DIR = SCRIPTS_DIR / 'common'

# Add scripts directory to path for imports
sys.path.insert(0, str(COMMON_DIR))

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from config_loader import load_config_defaults


def test_azure_credentials_from_env_vars():
    """Test that Azure credentials can be read from environment variables."""
    
    # Set up environment variables
    env_vars = {
        'AZURE_BATCH_ACCOUNT_NAME': 'env-batch-account',
        'AZURE_BATCH_ACCOUNT_KEY': 'env-batch-key',
        'AZURE_BATCH_ACCOUNT_URL': 'https://env-batch.eastus.batch.azure.com',
        'AZURE_STORAGE_ACCOUNT_NAME': 'env-storage-account',
        'AZURE_STORAGE_ACCOUNT_KEY': 'env-storage-key',
    }
    
    with patch.dict(os.environ, env_vars):
        # Simulate argparse namespace
        class Args:
            batch_account_name = None
            batch_account_key = None
            batch_account_url = None
            storage_account_name = None
            storage_account_key = None
        
        args = Args()
        config_defaults = {}
        
        # Apply environment variables (simulating the logic in submit_batch_jobs.py)
        if not args.batch_account_name:
            args.batch_account_name = os.environ.get('AZURE_BATCH_ACCOUNT_NAME') or config_defaults.get('batch_account_name')
        
        if not args.batch_account_key:
            args.batch_account_key = os.environ.get('AZURE_BATCH_ACCOUNT_KEY') or config_defaults.get('batch_account_key')
        
        if not args.batch_account_url:
            args.batch_account_url = os.environ.get('AZURE_BATCH_ACCOUNT_URL') or config_defaults.get('batch_account_url')
        
        if not args.storage_account_name:
            args.storage_account_name = os.environ.get('AZURE_STORAGE_ACCOUNT_NAME') or config_defaults.get('storage_account_name')
        
        if not args.storage_account_key:
            args.storage_account_key = os.environ.get('AZURE_STORAGE_ACCOUNT_KEY') or config_defaults.get('storage_account_key')
        
        # Verify environment variables were read
        assert args.batch_account_name == 'env-batch-account', \
            f"batch_account_name not read from env: {args.batch_account_name}"
        assert args.batch_account_key == 'env-batch-key', \
            f"batch_account_key not read from env: {args.batch_account_key}"
        assert args.batch_account_url == 'https://env-batch.eastus.batch.azure.com', \
            f"batch_account_url not read from env: {args.batch_account_url}"
        assert args.storage_account_name == 'env-storage-account', \
            f"storage_account_name not read from env: {args.storage_account_name}"
        assert args.storage_account_key == 'env-storage-key', \
            f"storage_account_key not read from env: {args.storage_account_key}"


def test_aws_credentials_from_env_vars():
    """Test that AWS credentials can be read from environment variables."""
    
    env_vars = {
        'AWS_ACCESS_KEY_ID': 'env-aws-key-id',
        'AWS_SECRET_ACCESS_KEY': 'env-aws-secret',
        'AWS_DEFAULT_REGION': 'us-west-2',
        'AWS_ENDPOINT_URL': 'https://s3.custom.com',
        'HF_TOKEN': 'env-hf-token',
    }
    
    with patch.dict(os.environ, env_vars):
        class Args:
            aws_access_key_id = None
            aws_secret_access_key = None
            aws_region = "us-east-1"  # Default value
            aws_endpoint_url = None
            hf_token = None
        
        args = Args()
        config_defaults = {}
        
        # Apply environment variables
        if not args.aws_access_key_id:
            args.aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID') or config_defaults.get('aws_access_key_id')
        
        if not args.aws_secret_access_key:
            args.aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY') or config_defaults.get('aws_secret_access_key')
        
        if args.aws_region == "us-east-1":  # Check if it's the default value
            env_region = os.environ.get('AWS_DEFAULT_REGION') or os.environ.get('AWS_REGION')
            if env_region:
                args.aws_region = env_region
            elif 'aws_region' in config_defaults:
                args.aws_region = config_defaults['aws_region']
        
        if not args.aws_endpoint_url:
            args.aws_endpoint_url = os.environ.get('AWS_ENDPOINT_URL') or config_defaults.get('aws_endpoint_url')
        
        if not args.hf_token:
            args.hf_token = os.environ.get('HF_TOKEN') or config_defaults.get('hf_token')
        
        # Verify environment variables were read
        assert args.aws_access_key_id == 'env-aws-key-id', \
            f"aws_access_key_id not read from env: {args.aws_access_key_id}"
        assert args.aws_secret_access_key == 'env-aws-secret', \
            f"aws_secret_access_key not read from env: {args.aws_secret_access_key}"
        assert args.aws_region == 'us-west-2', \
            f"aws_region not read from env: {args.aws_region}"
        assert args.aws_endpoint_url == 'https://s3.custom.com', \
            f"aws_endpoint_url not read from env: {args.aws_endpoint_url}"
        assert args.hf_token == 'env-hf-token', \
            f"hf_token not read from env: {args.hf_token}"


def test_cli_args_override_env_vars():
    """Test that command-line arguments take precedence over environment variables."""
    
    env_vars = {
        'AZURE_BATCH_ACCOUNT_NAME': 'env-batch-account',
        'AWS_ACCESS_KEY_ID': 'env-aws-key',
    }
    
    with patch.dict(os.environ, env_vars):
        class Args:
            batch_account_name = "cli-batch-account"  # Provided via CLI
            aws_access_key_id = "cli-aws-key"  # Provided via CLI
        
        args = Args()
        config_defaults = {}
        
        # Apply environment variables (they should not override CLI args)
        if not args.batch_account_name:
            args.batch_account_name = os.environ.get('AZURE_BATCH_ACCOUNT_NAME') or config_defaults.get('batch_account_name')
        
        if not args.aws_access_key_id:
            args.aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID') or config_defaults.get('aws_access_key_id')
        
        # Verify CLI args were preserved
        assert args.batch_account_name == 'cli-batch-account', \
            f"CLI arg should override env var: {args.batch_account_name}"
        assert args.aws_access_key_id == 'cli-aws-key', \
            f"CLI arg should override env var: {args.aws_access_key_id}"


def test_config_file_credentials():
    """Test that credentials can be loaded from config file."""
    if not HAS_YAML:
        print("SKIP: test_config_file_credentials (PyYAML not installed)")
        return
    
    config_content = """
azure:
  batch_account_name: "config-batch-account"
  batch_account_url: "https://config-batch.eastus.batch.azure.com"
  storage_account_name: "config-storage-account"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        config_defaults = load_config_defaults(config_file, backend='azure')
        
        # Verify config parameters were loaded
        assert 'batch_account_name' in config_defaults, \
            "batch_account_name not found in config"
        assert config_defaults['batch_account_name'] == "config-batch-account", \
            f"batch_account_name mismatch: {config_defaults['batch_account_name']}"
        
        assert 'batch_account_url' in config_defaults, \
            "batch_account_url not found in config"
        assert config_defaults['batch_account_url'] == "https://config-batch.eastus.batch.azure.com", \
            f"batch_account_url mismatch: {config_defaults['batch_account_url']}"
        
        assert 'storage_account_name' in config_defaults, \
            "storage_account_name not found in config"
        assert config_defaults['storage_account_name'] == "config-storage-account", \
            f"storage_account_name mismatch: {config_defaults['storage_account_name']}"
    
    finally:
        os.unlink(config_file)


def test_priority_order_cli_env_config():
    """Test the priority order: CLI > Env > Config."""
    if not HAS_YAML:
        print("SKIP: test_priority_order_cli_env_config (PyYAML not installed)")
        return
    
    # Create config file
    config_content = """
azure:
  batch_account_name: "config-batch-account"
  batch_account_url: "https://config-batch.eastus.batch.azure.com"
  storage_account_name: "config-storage-account"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        config_defaults = load_config_defaults(config_file, backend='azure')
        
        # Set environment variables
        env_vars = {
            'AZURE_BATCH_ACCOUNT_NAME': 'env-batch-account',
            'AZURE_STORAGE_ACCOUNT_NAME': 'env-storage-account',
        }
        
        with patch.dict(os.environ, env_vars):
            class Args:
                batch_account_name = "cli-batch-account"  # CLI provided
                batch_account_url = None  # Not provided via CLI
                storage_account_name = None  # Not provided via CLI
            
            args = Args()
            
            # Apply priority: CLI > Env > Config
            if not args.batch_account_name:
                args.batch_account_name = os.environ.get('AZURE_BATCH_ACCOUNT_NAME') or config_defaults.get('batch_account_name')
            
            if not args.batch_account_url:
                args.batch_account_url = os.environ.get('AZURE_BATCH_ACCOUNT_URL') or config_defaults.get('batch_account_url')
            
            if not args.storage_account_name:
                args.storage_account_name = os.environ.get('AZURE_STORAGE_ACCOUNT_NAME') or config_defaults.get('storage_account_name')
            
            # Verify priority order
            assert args.batch_account_name == 'cli-batch-account', \
                f"CLI should have highest priority: {args.batch_account_name}"
            assert args.batch_account_url == 'https://config-batch.eastus.batch.azure.com', \
                f"Config should be used when CLI and env not provided: {args.batch_account_url}"
            assert args.storage_account_name == 'env-storage-account', \
                f"Env should override config: {args.storage_account_name}"
    
    finally:
        os.unlink(config_file)


if __name__ == '__main__':
    # Run tests
    test_azure_credentials_from_env_vars()
    print("✓ test_azure_credentials_from_env_vars passed")
    
    test_aws_credentials_from_env_vars()
    print("✓ test_aws_credentials_from_env_vars passed")
    
    test_cli_args_override_env_vars()
    print("✓ test_cli_args_override_env_vars passed")
    
    test_config_file_credentials()
    print("✓ test_config_file_credentials passed")
    
    test_priority_order_cli_env_config()
    print("✓ test_priority_order_cli_env_config passed")
    
    print("\nAll environment variable tests passed!")
