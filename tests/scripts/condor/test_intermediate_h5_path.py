#!/usr/bin/env python3
"""
Test intermediate_h5_path handling in HTCondor submission script.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts" / "condor"
sys.path.insert(0, str(scripts_dir))


def test_intermediate_h5_path_empty_string_from_config():
    """Test that empty string intermediate_h5_path from config is treated as None."""
    from submit_condor_jobs import CondorJobSubmitter
    
    submitter = CondorJobSubmitter()
    
    # Create a temporary config file with empty string for intermediate_h5_path
    config = {
        "defaults": {
            "prefilter_model_type": "CTRANSPATH",
            "intermediate_h5_path": ""  # Empty string from config
        },
        "tasks": [
            {
                "task_id": "test_task",
                "slide_path": "/path/to/slide.svs",
                "output_h5_path": "/path/to/output.h5",
                "output_pt_path": "/path/to/output.pt"
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_file = f.name
    
    try:
        # Mock submit_task to capture parameters
        captured_params = []
        def mock_submit_task(**kwargs):
            captured_params.append(kwargs)
            return "test_job_id"
        
        submitter.submit_task = mock_submit_task
        
        # Submit tasks from config
        job_ids = submitter.submit_tasks_from_config(config_file)
        
        # Verify intermediate_h5_path was normalized to None
        assert len(captured_params) == 1, "Expected 1 task to be submitted"
        params = captured_params[0]
        
        # Empty string should be normalized to None
        intermediate_path = params.get('intermediate_h5_path')
        assert intermediate_path is None, \
            f"intermediate_h5_path should be None when empty string is provided: {intermediate_path}"
        
        print("✓ Empty string intermediate_h5_path normalized to None in HTCondor config")
        
    finally:
        os.unlink(config_file)


if __name__ == '__main__':
    test_intermediate_h5_path_empty_string_from_config()
    print("\nAll HTCondor intermediate_h5_path tests passed!")
