#!/usr/bin/env python3
"""
Test for intermediate_h5_path handling in Azure Batch submission.
This test validates that intermediate_h5_path is only set when aggregation is used.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Get absolute paths using pathlib
TEST_DIR = Path(__file__).parent
SCRIPTS_DIR = TEST_DIR / '..' / '..' / '..' / 'scripts' / 'azure_batch'

# Add scripts directory to path for imports
sys.path.insert(0, str(SCRIPTS_DIR))

def test_intermediate_h5_path_with_aggregation():
    """Test that intermediate_h5_path is set when aggregation_method is specified."""
    from submit_batch_jobs import AzureBatchJobSubmitter
    
    # Mock Azure Batch client
    with patch('submit_batch_jobs.BatchServiceClient'):
        submitter = AzureBatchJobSubmitter(
            batch_account_name="test",
            batch_account_key="test",
            batch_account_url="https://test.batch.azure.com"
        )
        
        # Test CSV submission with aggregation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("slide_id,slide_path\n")
            f.write("slide1,/path/to/slide1.svs\n")
            csv_file = f.name
        
        try:
            # Mock submit_task to capture parameters
            captured_params = []
            def mock_submit_task(**kwargs):
                captured_params.append(kwargs)
            
            submitter.submit_task = mock_submit_task
            
            # Submit with aggregation method
            submitter.submit_tasks_from_csv(
                job_id="test-job",
                csv_file=csv_file,
                output_s3_prefix="s3://bucket/output",
                aggregation_method="model"  # This should trigger intermediate_h5_path
            )
            
            # Verify intermediate_h5_path was set
            assert len(captured_params) == 1, "Expected 1 task to be submitted"
            params = captured_params[0]
            
            # Check that intermediate_h5_path is present and not None
            assert 'intermediate_h5_path' in params, \
                "intermediate_h5_path should be present when aggregation_method is specified"
            assert params['intermediate_h5_path'] is not None, \
                "intermediate_h5_path should not be None when aggregation_method is specified"
            assert 'tile_h5' in params['intermediate_h5_path'], \
                f"intermediate_h5_path should contain 'tile_h5': {params['intermediate_h5_path']}"
            
            print(f"✓ With aggregation: intermediate_h5_path = {params['intermediate_h5_path']}")
            
        finally:
            os.unlink(csv_file)


def test_intermediate_h5_path_without_aggregation():
    """Test that intermediate_h5_path is NOT set when aggregation_method is not specified."""
    from submit_batch_jobs import AzureBatchJobSubmitter
    
    # Mock Azure Batch client
    with patch('submit_batch_jobs.BatchServiceClient'):
        submitter = AzureBatchJobSubmitter(
            batch_account_name="test",
            batch_account_key="test",
            batch_account_url="https://test.batch.azure.com"
        )
        
        # Test CSV submission without aggregation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("slide_id,slide_path\n")
            f.write("slide1,/path/to/slide1.svs\n")
            csv_file = f.name
        
        try:
            # Mock submit_task to capture parameters
            captured_params = []
            def mock_submit_task(**kwargs):
                captured_params.append(kwargs)
            
            submitter.submit_task = mock_submit_task
            
            # Submit without aggregation method
            submitter.submit_tasks_from_csv(
                job_id="test-job",
                csv_file=csv_file,
                output_s3_prefix="s3://bucket/output"
                # No aggregation_method specified - default is "identity"
            )
            
            # Verify intermediate_h5_path was NOT set
            assert len(captured_params) == 1, "Expected 1 task to be submitted"
            params = captured_params[0]
            
            # intermediate_h5_path should either be None or not present
            intermediate_path = params.get('intermediate_h5_path')
            assert intermediate_path is None, \
                f"intermediate_h5_path should be None when no aggregation: {intermediate_path}"
            
            print("✓ Without aggregation: intermediate_h5_path = None")
            
        finally:
            os.unlink(csv_file)


def test_intermediate_h5_path_with_identity_aggregation():
    """Test that intermediate_h5_path is NOT set when aggregation_method is 'identity'."""
    from submit_batch_jobs import AzureBatchJobSubmitter
    
    # Mock Azure Batch client
    with patch('submit_batch_jobs.BatchServiceClient'):
        submitter = AzureBatchJobSubmitter(
            batch_account_name="test",
            batch_account_key="test",
            batch_account_url="https://test.batch.azure.com"
        )
        
        # Test CSV submission with identity aggregation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("slide_id,slide_path\n")
            f.write("slide1,/path/to/slide1.svs\n")
            csv_file = f.name
        
        try:
            # Mock submit_task to capture parameters
            captured_params = []
            def mock_submit_task(**kwargs):
                captured_params.append(kwargs)
            
            submitter.submit_task = mock_submit_task
            
            # Submit with identity aggregation (no tile-level features needed)
            submitter.submit_tasks_from_csv(
                job_id="test-job",
                csv_file=csv_file,
                output_s3_prefix="s3://bucket/output",
                aggregation_method="identity"  # identity doesn't use intermediate files
            )
            
            # Verify intermediate_h5_path was NOT set
            assert len(captured_params) == 1, "Expected 1 task to be submitted"
            params = captured_params[0]
            
            # intermediate_h5_path should be None for identity aggregation
            intermediate_path = params.get('intermediate_h5_path')
            assert intermediate_path is None, \
                f"intermediate_h5_path should be None for identity aggregation: {intermediate_path}"
            
            print("✓ With identity aggregation: intermediate_h5_path = None")
            
        finally:
            os.unlink(csv_file)


def test_intermediate_h5_path_stage_and_submit():
    """Test intermediate_h5_path handling in stage_and_submit_tasks_from_csv."""
    from submit_batch_jobs import AzureBatchJobSubmitter
    
    # Mock Azure Batch client and Azure Files staging
    with patch('submit_batch_jobs.BatchServiceClient'), \
         patch('submit_batch_jobs.AzureFilesStaging'):
        
        submitter = AzureBatchJobSubmitter(
            batch_account_name="test",
            batch_account_key="test",
            batch_account_url="https://test.batch.azure.com",
            storage_account_name="teststorage",
            storage_account_key="testkey",
            azure_files_share_name="testshare"
        )
        
        # Mock the staging methods
        submitter.azure_files_staging = Mock()
        submitter.azure_files_staging.upload_file = Mock()
        
        # Test CSV with aggregation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("slide_id,slide_path\n")
            f.write("slide1,/path/to/slide1.svs\n")
            csv_file = f.name
        
        try:
            # Mock submit_task to capture parameters
            captured_params = []
            def mock_submit_task(**kwargs):
                captured_params.append(kwargs)
            
            submitter.submit_task = mock_submit_task
            
            # Test with aggregation
            submitter.stage_and_submit_tasks_from_csv(
                job_id="test-job",
                csv_file=csv_file,
                output_s3_prefix="s3://bucket/output",
                aggregation_method="model"
            )
            
            # Verify intermediate_h5_path was set
            assert len(captured_params) == 1
            params = captured_params[0]
            assert params.get('intermediate_h5_path') is not None, \
                "intermediate_h5_path should be set with aggregation"
            
            print(f"✓ Stage and submit with aggregation: intermediate_h5_path = {params['intermediate_h5_path']}")
            
            # Clear and test without aggregation
            captured_params.clear()
            
            submitter.stage_and_submit_tasks_from_csv(
                job_id="test-job",
                csv_file=csv_file,
                output_s3_prefix="s3://bucket/output"
                # No aggregation
            )
            
            # Verify intermediate_h5_path was NOT set
            assert len(captured_params) == 1
            params = captured_params[0]
            assert params.get('intermediate_h5_path') is None, \
                "intermediate_h5_path should be None without aggregation"
            
            print("✓ Stage and submit without aggregation: intermediate_h5_path = None")
            
        finally:
            os.unlink(csv_file)


if __name__ == '__main__':
    # Run tests
    test_intermediate_h5_path_with_aggregation()
    print()
    
    test_intermediate_h5_path_without_aggregation()
    print()
    
    test_intermediate_h5_path_with_identity_aggregation()
    print()
    
    test_intermediate_h5_path_stage_and_submit()
    print()
    
    print("All intermediate_h5_path tests passed!")
