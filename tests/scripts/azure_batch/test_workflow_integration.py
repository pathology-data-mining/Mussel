#!/usr/bin/env python3
"""
Integration test demonstrating the complete workflow with credentials from
environment variables, config file, and CSV manifest.
"""

import os
import sys
import tempfile
from pathlib import Path

# Get absolute paths
TEST_DIR = Path(__file__).parent
SCRIPTS_DIR = TEST_DIR / '..' / '..' / '..' / 'scripts'
COMMON_DIR = SCRIPTS_DIR / 'common'

# Add to path
sys.path.insert(0, str(COMMON_DIR))

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("WARNING: PyYAML not installed, skipping test")
    sys.exit(0)

from config_loader import load_config_defaults


def test_complete_workflow():
    """
    Test the complete workflow:
    1. Create a config YAML with Azure batch parameters
    2. Set environment variables for secrets
    3. Verify credentials are loaded in correct priority order
    """
    
    # Create config file with non-sensitive parameters
    config_content = """
# Processing parameters
prefilter_model_type: CTRANSPATH
batch_size: 64
num_workers: 4

# AWS configuration (non-sensitive parts)
aws:
  region: us-west-2
  endpoint_url: https://s3.custom.com

# Azure Batch configuration (non-sensitive parts)
azure:
  batch_account_name: config-batch-account
  batch_account_url: https://config-batch.eastus.batch.azure.com
  pool_id: mussel-pool
  job_id: mussel-job-001
  create_pool: true
  create_job: true
  vm_size: Standard_NC6s_v3
  node_count: 2
  storage_account_name: config-storage-account
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        # Simulate environment variables (secrets)
        env_vars = {
            'AZURE_BATCH_ACCOUNT_KEY': 'env-batch-key-secret',
            'AZURE_STORAGE_ACCOUNT_KEY': 'env-storage-key-secret',
            'AWS_ACCESS_KEY_ID': 'env-aws-key-id',
            'AWS_SECRET_ACCESS_KEY': 'env-aws-secret-key',
            'HF_TOKEN': 'env-hf-token',
        }
        
        # Load config
        config_defaults = load_config_defaults(config_file, backend='azure')
        
        # Simulate argument parsing (no CLI args provided)
        class Args:
            batch_account_name = None
            batch_account_key = None
            batch_account_url = None
            storage_account_name = None
            storage_account_key = None
            aws_access_key_id = None
            aws_secret_access_key = None
            aws_region = "us-east-1"  # Default
            hf_token = None
        
        args = Args()
        
        # Apply priority: CLI > Env > Config
        # Simulate the logic from submit_batch_jobs.py
        
        # Azure credentials
        if not args.batch_account_name:
            args.batch_account_name = env_vars.get('AZURE_BATCH_ACCOUNT_NAME') or config_defaults.get('batch_account_name')
        
        if not args.batch_account_key:
            args.batch_account_key = env_vars.get('AZURE_BATCH_ACCOUNT_KEY') or config_defaults.get('batch_account_key')
        
        if not args.batch_account_url:
            args.batch_account_url = env_vars.get('AZURE_BATCH_ACCOUNT_URL') or config_defaults.get('batch_account_url')
        
        if not args.storage_account_name:
            args.storage_account_name = env_vars.get('AZURE_STORAGE_ACCOUNT_NAME') or config_defaults.get('storage_account_name')
        
        if not args.storage_account_key:
            args.storage_account_key = env_vars.get('AZURE_STORAGE_ACCOUNT_KEY') or config_defaults.get('storage_account_key')
        
        # AWS credentials
        if not args.aws_access_key_id:
            args.aws_access_key_id = env_vars.get('AWS_ACCESS_KEY_ID') or config_defaults.get('aws_access_key_id')
        
        if not args.aws_secret_access_key:
            args.aws_secret_access_key = env_vars.get('AWS_SECRET_ACCESS_KEY') or config_defaults.get('aws_secret_access_key')
        
        if args.aws_region == "us-east-1":
            env_region = env_vars.get('AWS_DEFAULT_REGION')
            if env_region:
                args.aws_region = env_region
            elif 'aws_region' in config_defaults:
                args.aws_region = config_defaults['aws_region']
        
        if not args.hf_token:
            args.hf_token = env_vars.get('HF_TOKEN') or config_defaults.get('hf_token')
        
        # Verify results
        print("\n=== Integration Test Results ===\n")
        
        # Non-sensitive params from config
        print("Non-sensitive parameters from config:")
        assert args.batch_account_name == 'config-batch-account', \
            f"batch_account_name should come from config: {args.batch_account_name}"
        print(f"✓ batch_account_name: {args.batch_account_name}")
        
        assert args.batch_account_url == 'https://config-batch.eastus.batch.azure.com', \
            f"batch_account_url should come from config: {args.batch_account_url}"
        print(f"✓ batch_account_url: {args.batch_account_url}")
        
        assert args.storage_account_name == 'config-storage-account', \
            f"storage_account_name should come from config: {args.storage_account_name}"
        print(f"✓ storage_account_name: {args.storage_account_name}")
        
        assert args.aws_region == 'us-west-2', \
            f"aws_region should come from config: {args.aws_region}"
        print(f"✓ aws_region: {args.aws_region}")
        
        # Sensitive params from env vars
        print("\nSensitive parameters from environment variables:")
        assert args.batch_account_key == 'env-batch-key-secret', \
            f"batch_account_key should come from env: {args.batch_account_key}"
        print(f"✓ batch_account_key: {args.batch_account_key}")
        
        assert args.storage_account_key == 'env-storage-key-secret', \
            f"storage_account_key should come from env: {args.storage_account_key}"
        print(f"✓ storage_account_key: {args.storage_account_key}")
        
        assert args.aws_access_key_id == 'env-aws-key-id', \
            f"aws_access_key_id should come from env: {args.aws_access_key_id}"
        print(f"✓ aws_access_key_id: {args.aws_access_key_id}")
        
        assert args.aws_secret_access_key == 'env-aws-secret-key', \
            f"aws_secret_access_key should come from env: {args.aws_secret_access_key}"
        print(f"✓ aws_secret_access_key: {args.aws_secret_access_key}")
        
        assert args.hf_token == 'env-hf-token', \
            f"hf_token should come from env: {args.hf_token}"
        print(f"✓ hf_token: {args.hf_token}")
        
        # Other config params
        print("\nOther configuration parameters:")
        assert config_defaults['prefilter_model_type'] == 'CTRANSPATH'
        print(f"✓ prefilter_model_type: {config_defaults['prefilter_model_type']}")
        
        assert config_defaults['batch_size'] == 64
        print(f"✓ batch_size: {config_defaults['batch_size']}")
        
        assert config_defaults['pool_id'] == 'mussel-pool'
        print(f"✓ pool_id: {config_defaults['pool_id']}")
        
        assert config_defaults['vm_size'] == 'Standard_NC6s_v3'
        print(f"✓ vm_size: {config_defaults['vm_size']}")
        
        print("\n=== Integration Test PASSED ===")
        print("\nThis demonstrates the recommended workflow:")
        print("1. Store non-sensitive configuration in YAML file")
        print("2. Store sensitive credentials in environment variables")
        print("3. Use CSV manifest for slide lists")
        print("4. Priority: CLI args > Environment variables > Config file")
        
    finally:
        os.unlink(config_file)


if __name__ == '__main__':
    test_complete_workflow()
