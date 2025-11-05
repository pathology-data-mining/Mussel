#!/usr/bin/env python3
"""
Tests for Azure Batch job submission with Azure Files staging.
This test validates the logic and structure without requiring Azure SDK.
"""

import sys
import os
import re

# Add scripts path
scripts_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts', 'azure_batch')


def test_submit_batch_jobs_has_azure_files_support():
    """Test that submit_batch_jobs.py has Azure Files staging support."""
    submit_file = os.path.join(scripts_path, 'submit_batch_jobs.py')
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check for Azure Files staging import
    assert 'from azure_files_staging import AzureFilesStaging' in content, \
        "Missing azure_files_staging import"
    
    # Check for azure_files_share_name parameter in __init__
    assert 'azure_files_share_name: Optional[str] = None' in content, \
        "Missing azure_files_share_name parameter in __init__"
    
    # Check for stage_slides_to_azure_files method
    assert 'def stage_slides_to_azure_files(' in content, \
        "Missing stage_slides_to_azure_files method"
    
    # Check for cleanup_staged_files method
    assert 'def cleanup_staged_files(' in content, \
        "Missing cleanup_staged_files method"
    
    # Check for mount_azure_files parameter in create_pool
    assert 'mount_azure_files: bool = False' in content, \
        "Missing mount_azure_files parameter in create_pool"
    
    # Check for MountConfiguration setup
    assert 'batchmodels.MountConfiguration' in content, \
        "Missing MountConfiguration for Azure Files mount"
    
    # Check for staged_slide_paths parameter in submit_tasks_from_csv
    assert 'staged_slide_paths: Optional[Dict[str, str]] = None' in content, \
        "Missing staged_slide_paths parameter in submit_tasks_from_csv"
    
    # Check for command-line arguments
    assert '--azure-files-share-name' in content, \
        "Missing --azure-files-share-name argument"
    assert '--stage-to-azure-files' in content, \
        "Missing --stage-to-azure-files argument"
    assert '--mount-azure-files' in content, \
        "Missing --mount-azure-files argument"
    assert '--cleanup-staged-files' in content, \
        "Missing --cleanup-staged-files argument"
    
    print("✓ submit_batch_jobs.py has Azure Files staging support")


def test_run_script_has_azfiles_support():
    """Test that run_tessellate_extract_features.sh supports azfiles:// paths."""
    run_script = os.path.join(scripts_path, 'run_tessellate_extract_features.sh')
    
    with open(run_script, 'r') as f:
        content = f.read()
    
    # Check for is_azfiles_path function
    assert 'is_azfiles_path()' in content, \
        "Missing is_azfiles_path function"
    
    # Check for resolve_azfiles_path function
    assert 'resolve_azfiles_path()' in content, \
        "Missing resolve_azfiles_path function"
    
    # Check for azfiles path handling in staging section
    assert 'if is_azfiles_path "$SLIDE_PATH"' in content, \
        "Missing azfiles path handling for SLIDE_PATH"
    
    # Check that function resolves to mount point
    assert '/mnt/batch/tasks/fsmounts/azfiles' in content, \
        "Missing Azure Files mount point reference"
    
    print("✓ run_tessellate_extract_features.sh supports azfiles:// paths")


def test_azure_files_staging_module_structure():
    """Test that azure_files_staging.py has required functionality."""
    staging_file = os.path.join(scripts_path, '..', 'common', 'azure_files_staging.py')
    
    with open(staging_file, 'r') as f:
        content = f.read()
    
    # Check for AzureFilesStaging class
    assert 'class AzureFilesStaging:' in content, \
        "Missing AzureFilesStaging class"
    
    # Check for key methods
    assert 'def upload_file(' in content, \
        "Missing upload_file method"
    assert 'def delete_file(' in content, \
        "Missing delete_file method"
    assert 'def delete_directory(' in content, \
        "Missing delete_directory method"
    assert 'def cleanup_staging(' in content, \
        "Missing cleanup_staging method"
    
    # Check for helper functions
    assert 'def stage_slides_to_azure_files(' in content, \
        "Missing stage_slides_to_azure_files helper function"
    
    print("✓ azure_files_staging.py has required functionality")


def test_documentation_updated():
    """Test that documentation includes Azure Files staging information."""
    readme_file = os.path.join(scripts_path, 'README.md')
    
    with open(readme_file, 'r') as f:
        content = f.read()
    
    # Check for Azure Files section
    assert 'Azure Files Staging' in content or 'azure files' in content.lower(), \
        "Missing Azure Files staging section in documentation"
    
    # Check for key concepts
    assert '--stage-to-azure-files' in content, \
        "Missing --stage-to-azure-files documentation"
    assert '--mount-azure-files' in content, \
        "Missing --mount-azure-files documentation"
    assert '--cleanup-staged-files' in content, \
        "Missing --cleanup-staged-files documentation"
    
    # Check for workflow description
    assert 'azure-storage-file-share' in content, \
        "Missing azure-storage-file-share dependency in documentation"
    
    print("✓ Documentation includes Azure Files staging information")


def test_workflow_integration():
    """Test that the workflow is properly integrated."""
    submit_file = os.path.join(scripts_path, 'submit_batch_jobs.py')
    
    with open(submit_file, 'r') as f:
        content = f.read()
    
    # Check that staging happens before task submission
    # Look for pattern: stage_to_azure_files ... stage_slides_to_azure_files ... submit_tasks_from_csv
    stage_pattern = r'if args\.stage_to_azure_files:.*?staged_slide_paths = submitter\.stage_slides_to_azure_files'
    assert re.search(stage_pattern, content, re.DOTALL), \
        "Staging not properly integrated before task submission"
    
    # Check that staged paths are passed to submit_tasks_from_csv
    assert 'staged_slide_paths=staged_slide_paths' in content, \
        "Staged paths not passed to submit_tasks_from_csv"
    
    # Check that cleanup is called
    assert 'if args.cleanup_staged_files:' in content, \
        "Cleanup not properly integrated"
    assert 'submitter.cleanup_staged_files()' in content, \
        "cleanup_staged_files not called"
    
    print("✓ Workflow is properly integrated")


if __name__ == "__main__":
    print("\nRunning Azure Batch Azure Files staging validation tests...\n")
    
    try:
        test_submit_batch_jobs_has_azure_files_support()
        test_run_script_has_azfiles_support()
        test_azure_files_staging_module_structure()
        test_documentation_updated()
        test_workflow_integration()
        
        print("\n✅ All validation tests passed!\n")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running tests: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

