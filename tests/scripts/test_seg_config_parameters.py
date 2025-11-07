#!/usr/bin/env python3
"""
Tests for SegConfig parameter support in distributed batch scripts.
This test validates that all SegConfig parameters and groups are properly configured
in SLURM, HTCondor, and Azure Batch submission scripts.
"""

from pathlib import Path


# Get absolute paths using pathlib
TEST_DIR = Path(__file__).parent
SCRIPTS_DIR = TEST_DIR / '..' / '..' / 'scripts'
SLURM_SCRIPT = SCRIPTS_DIR / 'slurm' / 'submit_slurm_jobs.py'
CONDOR_SCRIPT = SCRIPTS_DIR / 'condor' / 'submit_condor_jobs.py'
AZURE_SCRIPT = SCRIPTS_DIR / 'azure_batch' / 'submit_batch_jobs.py'
RUN_SCRIPT = SCRIPTS_DIR / 'common' / 'run_tessellate_extract_features.sh'


def test_seg_config_group_parameter_slurm():
    """Test that --seg-config-group parameter is defined in SLURM script."""
    with open(SLURM_SCRIPT, 'r') as f:
        content = f.read()
    
    assert '--seg-config-group' in content, \
        "Missing --seg-config-group CLI parameter in SLURM script"
    
    # Check for the choices (biopsy, resection, tcga)
    assert 'biopsy' in content and 'resection' in content and 'tcga' in content, \
        "Missing SegConfig group choices in SLURM script"


def test_seg_config_group_parameter_condor():
    """Test that --seg-config-group parameter is defined in HTCondor script."""
    with open(CONDOR_SCRIPT, 'r') as f:
        content = f.read()
    
    assert '--seg-config-group' in content, \
        "Missing --seg-config-group CLI parameter in HTCondor script"
    
    # Check for the choices (biopsy, resection, tcga)
    assert 'biopsy' in content and 'resection' in content and 'tcga' in content, \
        "Missing SegConfig group choices in HTCondor script"


def test_seg_config_parameters_slurm():
    """Test that all SegConfig parameters are defined in SLURM script."""
    with open(SLURM_SCRIPT, 'r') as f:
        content = f.read()
    
    # Check for all new SegConfig parameters
    required_params = [
        '--step-size',
        '--seg-level',
        '--segment-max-value',
        '--median-blur-ksize',
        '--morphology-ex-kernel',
        '--ref-patch-size',
        '--use-otsu',
        '--tissue-area-threshold',
        '--hole-area-threshold',
        '--max-num-holes',
        '--keep-ids',
        '--exclude-ids',
    ]
    
    for param in required_params:
        assert param in content, \
            f"Missing {param} CLI parameter in SLURM script"


def test_seg_config_parameters_condor():
    """Test that all SegConfig parameters are defined in HTCondor script."""
    with open(CONDOR_SCRIPT, 'r') as f:
        content = f.read()
    
    # Check for all new SegConfig parameters
    required_params = [
        '--step-size',
        '--seg-level',
        '--segment-max-value',
        '--median-blur-ksize',
        '--morphology-ex-kernel',
        '--ref-patch-size',
        '--use-otsu',
        '--tissue-area-threshold',
        '--hole-area-threshold',
        '--max-num-holes',
        '--keep-ids',
        '--exclude-ids',
    ]
    
    for param in required_params:
        assert param in content, \
            f"Missing {param} CLI parameter in HTCondor script"


def test_seg_config_in_method_signatures_slurm():
    """Test that generate_batch_script has all SegConfig parameters in SLURM script."""
    with open(SLURM_SCRIPT, 'r') as f:
        content = f.read()
    
    # Check that method signature includes new parameters
    assert 'seg_config_group: Optional[str] = None' in content, \
        "seg_config_group parameter missing in generate_batch_script signature"
    
    assert 'step_size: Optional[int] = None' in content, \
        "step_size parameter missing in generate_batch_script signature"
    
    assert 'median_blur_ksize: Optional[int] = None' in content, \
        "median_blur_ksize parameter missing in generate_batch_script signature"


def test_seg_config_in_method_signatures_condor():
    """Test that generate_submit_file has all SegConfig parameters in HTCondor script."""
    with open(CONDOR_SCRIPT, 'r') as f:
        content = f.read()
    
    # Check that method signature includes new parameters
    assert 'seg_config_group: Optional[str] = None' in content, \
        "seg_config_group parameter missing in generate_submit_file signature"
    
    assert 'step_size: Optional[int] = None' in content, \
        "step_size parameter missing in generate_submit_file signature"
    
    assert 'median_blur_ksize: Optional[int] = None' in content, \
        "median_blur_ksize parameter missing in generate_submit_file signature"


def test_seg_config_in_method_signatures_azure():
    """Test that submit_task has all SegConfig parameters in Azure Batch script."""
    with open(AZURE_SCRIPT, 'r') as f:
        content = f.read()
    
    # Check that method signature includes new parameters
    assert 'seg_config_group: Optional[str] = None' in content, \
        "seg_config_group parameter missing in submit_task signature"
    
    assert 'step_size: Optional[int] = None' in content, \
        "step_size parameter missing in submit_task signature"
    
    assert 'median_blur_ksize: Optional[int] = None' in content, \
        "median_blur_ksize parameter missing in submit_task signature"


def test_seg_config_env_vars_slurm():
    """Test that SegConfig parameters are passed as environment variables in SLURM script."""
    with open(SLURM_SCRIPT, 'r') as f:
        content = f.read()
    
    # Check for environment variable assignments
    assert 'SEG_CONFIG_GROUP' in content, \
        "SEG_CONFIG_GROUP environment variable not set in SLURM script"
    
    assert 'STEP_SIZE' in content, \
        "STEP_SIZE environment variable not set in SLURM script"
    
    assert 'MEDIAN_BLUR_KSIZE' in content, \
        "MEDIAN_BLUR_KSIZE environment variable not set in SLURM script"


def test_seg_config_env_vars_condor():
    """Test that SegConfig parameters are passed as environment variables in HTCondor script."""
    with open(CONDOR_SCRIPT, 'r') as f:
        content = f.read()
    
    # Check for environment variable assignments
    assert 'SEG_CONFIG_GROUP' in content, \
        "SEG_CONFIG_GROUP environment variable not set in HTCondor script"
    
    assert 'STEP_SIZE' in content, \
        "STEP_SIZE environment variable not set in HTCondor script"
    
    assert 'MEDIAN_BLUR_KSIZE' in content, \
        "MEDIAN_BLUR_KSIZE environment variable not set in HTCondor script"


def test_seg_config_env_vars_azure():
    """Test that SegConfig parameters are passed as environment variables in Azure Batch script."""
    with open(AZURE_SCRIPT, 'r') as f:
        content = f.read()
    
    # Check for environment variable assignments
    assert 'SEG_CONFIG_GROUP' in content, \
        "SEG_CONFIG_GROUP environment variable not set in Azure Batch script"
    
    assert 'STEP_SIZE' in content, \
        "STEP_SIZE environment variable not set in Azure Batch script"
    
    assert 'MEDIAN_BLUR_KSIZE' in content, \
        "MEDIAN_BLUR_KSIZE environment variable not set in Azure Batch script"


def test_run_script_seg_config_group():
    """Test that run script handles SEG_CONFIG_GROUP environment variable."""
    with open(RUN_SCRIPT, 'r') as f:
        content = f.read()
    
    # Check that run script handles SEG_CONFIG_GROUP
    assert 'SEG_CONFIG_GROUP' in content, \
        "SEG_CONFIG_GROUP not handled in run script"
    
    # Check that it uses Hydra's seg_config group syntax
    assert 'seg_config=' in content, \
        "seg_config group syntax not used in run script"


def test_run_script_seg_config_parameters():
    """Test that run script passes all SegConfig parameters to CLI."""
    with open(RUN_SCRIPT, 'r') as f:
        content = f.read()
    
    # Check for environment variable usage
    env_vars = [
        'STEP_SIZE',
        'SEG_LEVEL',
        'SEGMENT_MAX_VALUE',
        'MEDIAN_BLUR_KSIZE',
        'MORPHOLOGY_EX_KERNEL',
        'REF_PATCH_SIZE',
        'USE_OTSU',
        'TISSUE_AREA_THRESHOLD',
        'HOLE_AREA_THRESHOLD',
        'MAX_NUM_HOLES',
        'KEEP_IDS',
        'EXCLUDE_IDS',
    ]
    
    for env_var in env_vars:
        assert env_var in content, \
            f"{env_var} environment variable not handled in run script"


def test_run_script_conditional_seg_config():
    """Test that run script only passes SegConfig params when not using group."""
    with open(RUN_SCRIPT, 'r') as f:
        content = f.read()
    
    # Check that there's logic to skip individual params when group is set
    assert 'if [ -z "$SEG_CONFIG_GROUP" ]' in content, \
        "Missing conditional logic to skip individual params when group is set"


if __name__ == '__main__':
    # Run tests
    tests = [
        ("test_seg_config_group_parameter_slurm", test_seg_config_group_parameter_slurm),
        ("test_seg_config_group_parameter_condor", test_seg_config_group_parameter_condor),
        ("test_seg_config_parameters_slurm", test_seg_config_parameters_slurm),
        ("test_seg_config_parameters_condor", test_seg_config_parameters_condor),
        ("test_seg_config_in_method_signatures_slurm", test_seg_config_in_method_signatures_slurm),
        ("test_seg_config_in_method_signatures_condor", test_seg_config_in_method_signatures_condor),
        ("test_seg_config_in_method_signatures_azure", test_seg_config_in_method_signatures_azure),
        ("test_seg_config_env_vars_slurm", test_seg_config_env_vars_slurm),
        ("test_seg_config_env_vars_condor", test_seg_config_env_vars_condor),
        ("test_seg_config_env_vars_azure", test_seg_config_env_vars_azure),
        ("test_run_script_seg_config_group", test_run_script_seg_config_group),
        ("test_run_script_seg_config_parameters", test_run_script_seg_config_parameters),
        ("test_run_script_conditional_seg_config", test_run_script_conditional_seg_config),
    ]
    
    failed = []
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✓ {test_name} passed")
        except AssertionError as e:
            print(f"✗ {test_name} failed: {e}")
            failed.append((test_name, str(e)))
    
    if failed:
        print(f"\n{len(failed)} test(s) failed:")
        for test_name, error in failed:
            print(f"  - {test_name}: {error}")
        exit(1)
    else:
        print(f"\nAll {len(tests)} tests passed!")
