# Azure Files Staging Status
================================
Date: November 10, 2025 (15:46 UTC)

## Process Started

**PID**: 3269511 (bash) + 3269515 (python)
**Started**: 10:46 AM  
**Command**: Stage all 43,423 slides from revision_samples_with_paths.csv

## Configuration

- **Source**: S3 paths from revision_samples_with_paths.csv
- **S3 Endpoint**: http://pmindecs.mskcc.org:9020
- **Destination**: Azure Files share `mussel-staging`
- **Remote directory**: `revision_slides/`
- **Storage account**: mskpdmgen2

## Test Results

bœ… **5 slides staged successfully** (test run)
- 1106318.svs (P-0000012-T04-IM6)
- 881837.svs (P-0000034-T01-IM3)
- 755246.svs (P-0000037-T02-IM3)
- 1473555.svs (P-0000056-T01-IM3)
- 1376375.svs (P-0000058-T01-IM3)

## Estimate

**Total slides**: 43,423
**Average slide size**: ~200-500 MB
**Average upload time**: ~30-60 seconds per slide

**Estimated completion**:
- Optimistic (30s/slide): ~15 hours
- Realistic (45s/slide): ~22 hours  
- Pessimistic (60s/slide): ~30 hours

**Expected completion**: November 11, 2025 @ ~8-16:00 UTC

## Monitoring

Check progress:
```bash
# See current slide being processed
tail -f azure_full_staging.log

# Count how many slides have been staged
ls -1 /mnt/azfiles/revision_slides/ | wc -l

# Check process status
ps aux | grep 3269515
```

## Output

Final CSV with azfiles:// paths will be written to:
`revision_samples_staged_azfiles.csv`

Format:
```csv
image_id,sample_id,azfiles_path
1106318,P-0000012-T04-IM6,azfiles://mskpdmgen2/mussel-staging/revision_slides/1106318.svs
...
```

## Next Steps

After staging completes:
1. Use `revision_samples_staged_azfiles.csv` for batch processing
2. Slides will be accessed directly from Azure Files mount
3. No re-upload needed - use azfiles:// paths directly

## Script Details

**Script**: `scripts/azure_batch/stage_slides_to_azure_files.py`

Features:
- Downloads from S3 (handles custom endpoints)
- Uploads to Azure Files
- Resumes from any point (--resume-from N)
- Limits for testing (--limit N)
- Exports azfiles:// paths to CSV

## Troubleshooting

If staging fails:
1. Check process still running: `ps aux | grep 3269515`
2. Check log: `tail -100 azure_full_staging.log`
3. Resume from last slide: `--resume-from N` where N is last successful slide number
4. Check Azure Files capacity/quota

## Status: RUNNING âœ…

The staging process is actively running and will complete in 15-30 hours.
All slides from `mskmind-bkt/reef-slides/` will be staged to Azure Files.

