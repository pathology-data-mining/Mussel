# SLURM Job Submission - Quick Start Guide

## ✅ Test Status: SUCCESSFUL (Job 2627080)

## Quick Submit Command
```bash
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest test_slides.csv \
  --config slurm_test.yaml \
  --output-dir outputs/slurm_test \
  --submit
```

## Test Results Summary
- **Slides processed:** 9/9 ✅
- **Success rate:** 100%
- **Processing time:** 51s - 3min per slide
- **Output size:** 951 MB (18 files)
- **Model:** CTRANSPATH (768-dim features)

## Key Configuration (slurm_test.yaml)
```yaml
prefilter_model_type: CTRANSPATH
prefilter_model_path: /gpfs/mskmind_ess/limr/repos/TransPath/ctranspath.pth
patch_size: 224  # CTRANSPATH requirement
step_size: 224
batch_size: 64
num_workers: 8
use_gpu: true
seg_config:
  group: biopsy
```

## Important Fixes Applied
1. **CSV Parsing:** Added `\r\n` stripping for Windows line endings
2. **Patch Size:** Set to 224x224 for CTRANSPATH compatibility
3. **Model Compatibility:** Disabled classifier/OPTIMUS due to pickle issues

## Output Files
Each slide generates:
- `{slide_id}_features.h5` - HDF5 format (coords + features)
- `{slide_id}_features.pt` - PyTorch tensor (features only)

## Monitoring Jobs
```bash
# Check job status
squeue -j <job_id>

# View logs
tail -f slurm_logs/mussel_array_<job_id>_<task>.out
tail -f slurm_logs/mussel_array_<job_id>_<task>.err

# Check completed jobs
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed
```

## Dry Run (Test Without Submitting)
```bash
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest test_slides.csv \
  --config slurm_test.yaml \
  --output-dir outputs/slurm_test
  # No --submit flag = dry run
```

## Resource Allocation
- **CPUs:** 4 per task
- **Memory:** 16GB per task
- **GPUs:** 1 per task
- **Time limit:** 2 hours
- **Parallel jobs:** Up to 6 simultaneous

## Known Issues
- ❌ Classifier filtering: Pickle compatibility issue
- ❌ OPTIMUS postfilter: Pickle compatibility issue
- ✅ CTRANSPATH extraction: Working perfectly

## Next Steps for Production
1. Re-save models with current torch/sklearn versions
2. Test with larger slide sets
3. Enable classifier filtering once pickle issue resolved
4. Add automated result validation

## Files Modified
- `scripts/slurm/submit_slurm_jobs.py` (CSV parsing fix)
- `slurm_test.yaml` (patch size configuration)

See `slurm_test_summary.txt` for detailed test report.
