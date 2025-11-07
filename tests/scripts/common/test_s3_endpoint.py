#!/usr/bin/env python3
"""
Tests for custom S3 endpoint support in distributed processing scripts.
"""

import os
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Get absolute paths
TEST_DIR = Path(__file__).parent
SCRIPTS_COMMON_DIR = TEST_DIR / '..' / '..' / '..' / 'scripts' / 'common'

# Add scripts directory to path to import modules
sys.path.insert(0, str(SCRIPTS_COMMON_DIR))


def test_azure_files_staging_supports_custom_endpoint():
    """Test that azure_files_staging.py supports AWS_ENDPOINT_URL."""
    staging_file = SCRIPTS_COMMON_DIR / 'azure_files_staging.py'
    
    with open(staging_file, 'r') as f:
        content = f.read()
    
    # Check that the code references AWS_ENDPOINT_URL
    assert 'AWS_ENDPOINT_URL' in content, \
        "Missing AWS_ENDPOINT_URL environment variable support"
    
    # Check that endpoint-url flag is used
    assert '--endpoint-url' in content, \
        "Missing --endpoint-url flag for AWS CLI"
    
    # Check that it's part of the _download_from_s3 method
    assert 'def _download_from_s3' in content, \
        "Missing _download_from_s3 method"
    
    print("✓ azure_files_staging.py supports custom S3 endpoint")


def test_azure_files_staging_download_from_s3_with_custom_endpoint():
    """Test _download_from_s3 method with custom endpoint."""
    # Import after adding to path
    from azure_files_staging import AzureFilesStaging
    
    # Mock the Azure dependencies
    with patch('azure_files_staging.ShareServiceClient'):
        # Create staging instance (using mocks for Azure dependencies)
        staging = AzureFilesStaging.__new__(AzureFilesStaging)
        staging.account_name = "test_account"
        staging.account_key = "test_key"
        staging.share_name = "test_share"
        
        # Test with custom endpoint
        with patch.dict(os.environ, {'AWS_ENDPOINT_URL': 'http://minio.local:9000'}):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                with patch('tempfile.NamedTemporaryFile') as mock_temp:
                    mock_temp.return_value.name = '/tmp/test.txt'
                    mock_temp.return_value.close = MagicMock()
                    
                    result = staging._download_from_s3("s3://bucket/file.txt")
                    
                    # Verify the command includes endpoint-url
                    called_cmd = mock_run.call_args[0][0]
                    assert '--endpoint-url' in called_cmd, \
                        "AWS CLI command should include --endpoint-url"
                    assert 'http://minio.local:9000' in called_cmd, \
                        "AWS CLI command should include the endpoint URL"
                    # Verify correct ordering: --endpoint-url should come before the endpoint value
                    endpoint_idx = called_cmd.index('--endpoint-url')
                    url_idx = called_cmd.index('http://minio.local:9000')
                    assert url_idx == endpoint_idx + 1, \
                        "endpoint URL should immediately follow --endpoint-url flag"
        
        # Test without custom endpoint
        with patch.dict(os.environ, {}, clear=True):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                with patch('tempfile.NamedTemporaryFile') as mock_temp:
                    mock_temp.return_value.name = '/tmp/test.txt'
                    mock_temp.return_value.close = MagicMock()
                    
                    result = staging._download_from_s3("s3://bucket/file.txt")
                    
                    # Verify the command does NOT include endpoint-url
                    called_cmd = mock_run.call_args[0][0]
                    assert '--endpoint-url' not in called_cmd, \
                        "AWS CLI command should not include --endpoint-url when not set"
    
    print("✓ _download_from_s3 correctly handles custom endpoint")


def test_model_predownload_supports_custom_endpoint():
    """Test that model_predownload.py supports AWS_ENDPOINT_URL."""
    predownload_file = SCRIPTS_COMMON_DIR / 'model_predownload.py'
    
    with open(predownload_file, 'r') as f:
        content = f.read()
    
    # Check that the code references AWS_ENDPOINT_URL
    assert 'AWS_ENDPOINT_URL' in content, \
        "Missing AWS_ENDPOINT_URL environment variable support"
    
    # Check that endpoint_url is passed to boto3
    assert 'endpoint_url' in content, \
        "Missing endpoint_url parameter for boto3"
    
    # Check that it's part of upload_models_to_s3
    assert 'def upload_models_to_s3' in content, \
        "Missing upload_models_to_s3 function"
    
    print("✓ model_predownload.py supports custom S3 endpoint")


def test_model_predownload_upload_with_custom_endpoint():
    """Test upload_models_to_s3 function with custom endpoint."""
    from model_predownload import upload_models_to_s3
    
    cached_models = {
        'CTRANSPATH': '/tmp/ctranspath.pth',
        'CLIP': '/tmp/clip.pth'
    }
    
    # Test with custom endpoint
    with patch.dict(os.environ, {'AWS_ENDPOINT_URL': 'http://minio.local:9000'}):
        with patch('boto3.client') as mock_boto_client:
            mock_s3 = MagicMock()
            mock_boto_client.return_value = mock_s3
            
            result = upload_models_to_s3(cached_models, 's3://test-bucket/models/')
            
            # Verify boto3.client was called with endpoint_url
            mock_boto_client.assert_called_once_with('s3', endpoint_url='http://minio.local:9000')
            
            # Verify upload_file was called for each model
            assert mock_s3.upload_file.call_count == 2
    
    # Test without custom endpoint
    with patch.dict(os.environ, {}, clear=True):
        with patch('boto3.client') as mock_boto_client:
            mock_s3 = MagicMock()
            mock_boto_client.return_value = mock_s3
            
            result = upload_models_to_s3(cached_models, 's3://test-bucket/models/')
            
            # Verify boto3.client was called without endpoint_url
            mock_boto_client.assert_called_once_with('s3')
    
    print("✓ upload_models_to_s3 correctly handles custom endpoint")


def test_run_tessellate_extract_features_supports_custom_endpoint():
    """Test that run_tessellate_extract_features.sh supports AWS_ENDPOINT_URL."""
    script_file = SCRIPTS_COMMON_DIR / 'run_tessellate_extract_features.sh'
    
    with open(script_file, 'r') as f:
        content = f.read()
    
    # Check that AWS_ENDPOINT_URL is documented
    assert 'AWS_ENDPOINT_URL' in content, \
        "Missing AWS_ENDPOINT_URL in environment variable documentation"
    
    # Check that endpoint-url is used in download_from_s3
    assert '--endpoint-url' in content, \
        "Missing --endpoint-url in S3 functions"
    
    # Verify it's in both download and upload functions
    lines_with_endpoint = [line for line in content.split('\n') if '--endpoint-url' in line]
    assert len(lines_with_endpoint) >= 2, \
        "Should have --endpoint-url in both download_from_s3 and upload_to_s3"
    
    print("✓ run_tessellate_extract_features.sh supports custom S3 endpoint")


def test_azure_batch_script_supports_custom_endpoint():
    """Test that Azure Batch script supports AWS_ENDPOINT_URL."""
    script_file = TEST_DIR / '..' / '..' / '..' / 'scripts' / 'azure_batch' / 'run_tessellate_extract_features.sh'
    
    with open(script_file, 'r') as f:
        content = f.read()
    
    # Check that AWS_ENDPOINT_URL is documented
    assert 'AWS_ENDPOINT_URL' in content, \
        "Missing AWS_ENDPOINT_URL in environment variable documentation"
    
    # Check that endpoint-url is used
    assert '--endpoint-url' in content, \
        "Missing --endpoint-url in S3 functions"
    
    # Verify it's in both download and upload functions
    lines_with_endpoint = [line for line in content.split('\n') if '--endpoint-url' in line]
    assert len(lines_with_endpoint) >= 2, \
        "Should have --endpoint-url in both download_from_s3 and upload_to_s3"
    
    print("✓ Azure Batch script supports custom S3 endpoint")


if __name__ == "__main__":
    print("\nRunning custom S3 endpoint support tests...\n")
    
    try:
        test_azure_files_staging_supports_custom_endpoint()
        test_azure_files_staging_download_from_s3_with_custom_endpoint()
        test_model_predownload_supports_custom_endpoint()
        test_model_predownload_upload_with_custom_endpoint()
        test_run_tessellate_extract_features_supports_custom_endpoint()
        test_azure_batch_script_supports_custom_endpoint()
        
        print("\n✅ All custom S3 endpoint tests passed!\n")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running tests: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
