#!/usr/bin/env python3
"""
Tests for SLURM partition behavior.

This test verifies that when no partition is specified, the generated
sbatch script does not include the #SBATCH --partition directive.
"""

import os
import sys

import pytest

# Import the SLURM submitter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts', 'slurm'))
from submit_slurm_jobs import SlurmJobSubmitter


def test_partition_not_specified():
    """
    Test that when partition is None, the sbatch script does not include --partition directive.
    """
    submitter = SlurmJobSubmitter()
    
    # Generate batch script without partition
    batch_content = submitter.generate_batch_script(
        job_name="test_job",
        slide_path="/path/to/slide.svs",
        output_h5_path="/path/to/output.h5",
        output_pt_path="/path/to/output.pt",
        partition=None,
    )
    
    # Verify partition directive is NOT in the script
    assert "#SBATCH --partition=" not in batch_content
    # Verify other directives are still present
    assert "#SBATCH --job-name=test_job" in batch_content
    assert "#SBATCH --cpus-per-task=" in batch_content
    assert "#SBATCH --mem=" in batch_content
    assert "#SBATCH --time=" in batch_content


def test_partition_specified():
    """
    Test that when partition is specified, the sbatch script includes --partition directive.
    """
    submitter = SlurmJobSubmitter()
    
    # Generate batch script with partition
    batch_content = submitter.generate_batch_script(
        job_name="test_job",
        slide_path="/path/to/slide.svs",
        output_h5_path="/path/to/output.h5",
        output_pt_path="/path/to/output.pt",
        partition="gpu",
    )
    
    # Verify partition directive IS in the script
    assert "#SBATCH --partition=gpu" in batch_content
    # Verify other directives are still present
    assert "#SBATCH --job-name=test_job" in batch_content


def test_partition_empty_string():
    """
    Test that when partition is empty string, the sbatch script does not include --partition directive.
    """
    submitter = SlurmJobSubmitter()
    
    # Generate batch script with empty string partition
    batch_content = submitter.generate_batch_script(
        job_name="test_job",
        slide_path="/path/to/slide.svs",
        output_h5_path="/path/to/output.h5",
        output_pt_path="/path/to/output.pt",
        partition="",
    )
    
    # Verify partition directive is NOT in the script
    assert "#SBATCH --partition=" not in batch_content


def test_batch_array_partition_not_specified():
    """
    Test that job array submission without partition does not include --partition directive.
    """
    submitter = SlurmJobSubmitter()
    
    # Create test slides data
    slides = [
        {'slide_id': 'slide1', 'slide_path': '/path/to/slide1.svs'},
        {'slide_id': 'slide2', 'slide_path': '/path/to/slide2.svs'},
    ]
    
    # Generate array batch script without partition
    # We'll use the internal method to access the generated script
    import tempfile
    import os
    
    # Temporarily change directory to a temp dir for this test
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        
        try:
            # Call _submit_job_array with submit=False to generate files only
            submitter._submit_job_array(
                slides=slides,
                output_dir="/output",
                output_s3_prefix=None,
                submit=False,
                partition=None,  # No partition specified
            )
            
            # Read the generated batch script
            with open("slurm_array_job.sbatch", "r") as f:
                batch_content = f.read()
            
            # Verify partition directive is NOT in the script
            assert "#SBATCH --partition=" not in batch_content
            # Verify other directives are still present
            assert "#SBATCH --job-name=mussel_array" in batch_content
            assert "#SBATCH --array=" in batch_content
        finally:
            os.chdir(original_dir)


def test_batch_array_partition_specified():
    """
    Test that job array submission with partition includes --partition directive.
    """
    submitter = SlurmJobSubmitter()
    
    # Create test slides data
    slides = [
        {'slide_id': 'slide1', 'slide_path': '/path/to/slide1.svs'},
        {'slide_id': 'slide2', 'slide_path': '/path/to/slide2.svs'},
    ]
    
    # Generate array batch script with partition
    import tempfile
    import os
    
    # Temporarily change directory to a temp dir for this test
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        
        try:
            # Call _submit_job_array with submit=False to generate files only
            submitter._submit_job_array(
                slides=slides,
                output_dir="/output",
                output_s3_prefix=None,
                submit=False,
                partition="gpu",  # Partition specified
            )
            
            # Read the generated batch script
            with open("slurm_array_job.sbatch", "r") as f:
                batch_content = f.read()
            
            # Verify partition directive IS in the script
            assert "#SBATCH --partition=gpu" in batch_content
            # Verify other directives are still present
            assert "#SBATCH --job-name=mussel_array" in batch_content
            assert "#SBATCH --array=" in batch_content
        finally:
            os.chdir(original_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
