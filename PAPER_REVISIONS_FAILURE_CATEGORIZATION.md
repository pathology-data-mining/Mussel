# Paper Revisions Processing - Failure Categorization Report

## Executive Summary

Out of 43,423 expected slides:
- **Successfully Processed**: 43,290 (99.69%)
- **Partial Failures**: 108 (0.25%) - processed but some models failed
- **Missing/Not Processed**: 25 (0.06%) - not processed at all

---

## Missing Slides Breakdown (25 total)

### Category 1: In S3 Bucket But Not Processed (5 slides)

These slides **exist in the S3 bucket** but were not processed. They likely were not staged to Azure Blob Storage or were excluded from the processing manifest.

| Slide ID | Sample ID | S3 Path |
|----------|-----------|---------|
| 10120 | TCGA-DD-A73A-01 | s3://pathology/TCGA/ef7fcc3a-ae85-4753-8ee3-a8e999eee196/TCGA-DD-A73A-01Z-00-DX1.6B409977-53A6-4DD9-9763-2094E5B83942.svs |
| 10561 | TCGA-DD-AAC8-01 | s3://pathology/TCGA/168084ab-1bd0-4914-8e14-47eb5e851a20/TCGA-DD-AAC8-01Z-00-DX1.9B06A570-EE68-494D-8798-8375AB92F895.svs |
| 2443 | TCGA-DD-A73G-01 | s3://pathology/TCGA/ac5ffd5c-441d-4440-ac3b-848ba82f1814/TCGA-DD-A73G-01Z-00-DX1.F5AB3B95-268C-44DA-9D42-7A7B5E1D8516.svs |
| 2978 | TCGA-DD-A4NS-01 | s3://pathology/TCGA/da7b10d7-fae7-4ec3-b408-7c646a558eae/TCGA-DD-A4NS-01Z-00-DX1.693CDBAD-EB24-4781-9FB5-DC08E0C8D847.svs |
| 434 | TCGA-DD-A4NR-01 | s3://pathology/TCGA/5fcdab38-ff40-4842-8126-e4fc3e97ac73/TCGA-DD-A4NR-01Z-00-DX1.8C4F2D2B-8B71-40F3-BA87-9B6360C526D3.svs |

**Action Required**: Stage these 5 slides to Azure Blob Storage and reprocess.

---

### Category 2: Has Local Path (1 slide)

This slide has a local filesystem path and was not processed in the Azure batch.

| Slide ID | Sample ID | Local Path |
|----------|-----------|------------|
| 762 | TCGA-PL-A8LX-01 | /gpfs/mskmind_ess/limr/repos/mussel-nf/missing_tcga_slides/TCGA-PL-A8LX-01A-01-DX1.9646D69F-A764-4246-9243-67A63006DE96.svs |

**Action Required**: Process locally via SLURM.

---

### Category 3: No Path in Manifest (19 slides)

These slides are listed in the manifest but have **NO path specified**. They are **not available** in either S3 or local storage and **CANNOT be processed** without locating the original files.

| Slide ID | Sample ID | Status |
|----------|-----------|--------|
| 4342539 | P-0065465-T01-IM7 | Source data unavailable |
| 4960133 | P-0070341-T01-IM7 | Source data unavailable |
| 5123565 | P-0072198-T01-IM7 | Source data unavailable |
| 5239023 | P-0028888-T02-IM7 | Source data unavailable |
| 5251287 | P-0066338-T03-IM7 | Source data unavailable |
| 5293771 | P-0058724-T02-IM7 | Source data unavailable |
| 5408319 | P-0075912-T01-IM7 | Source data unavailable |
| 5644243 | P-0079445-T01-IM7 | Source data unavailable |
| 5784666 | P-0081153-T01-IM7 | Source data unavailable |
| 5900034 | P-0080180-T02-IM7 | Source data unavailable |
| 5924314 | P-0083091-T01-IM7 | Source data unavailable |
| 5930652 | P-0082761-T01-IM7 | Source data unavailable |
| 5996220 | P-0083691-T01-IM7 | Source data unavailable |
| 6029392 | P-0084233-T01-IM7 | Source data unavailable |
| 6312303 | P-0086871-T01-IM7 | Source data unavailable |
| 7425333 | P-0099481-T01-IM7 | Source data unavailable |
| 7452736 | P-0067780-T06-IM7 | Source data unavailable |
| 7667146 | P-0101658-T02-IM7 | Source data unavailable |
| 7851252 | P-0104005-T01-IM7 | Source data unavailable |

**Action Required**: Document as 'SOURCE DATA UNAVAILABLE'. Investigate with MSK data owners if these slides can be located.

---

## Partial Failures Analysis (108 slides)

Slides that were processed but failed for one or more models:

### Model-Specific Failure Breakdown

| Model | Failures | Percentage of Partial Failures |
|-------|----------|-------------------------------|
| **TITAN_SLIDE** | **108** | **100%** |
| CONCH1_5 | 59 | 54.6% |
| GIGAPATH | 4 | 3.7% |
| GIGAPATH_SLIDE | 4 | 3.7% |
| UNI2 | 3 | 2.8% |

### Key Finding

**TITAN_SLIDE failed on ALL 108 partially-failed slides.**

This indicates a **systematic issue** with the TITAN_SLIDE model, not isolated failures. The issue is likely:
- Model configuration problem
- Resource constraint (memory/GPU)
- Compatibility issue with certain slide types
- Bug in the TITAN_SLIDE processing code

---

## Root Cause Analysis

### Missing Slides

1. **S3 slides not processed (5)**:
   - Root cause: Staging issue - slides exist in S3 but were not included in Azure Blob staging or processing manifest
   - Likely they were added to the manifest after the staging was completed

2. **Local path slide (1)**:
   - Root cause: Available locally but excluded from Azure batch
   - This is a TCGA slide that was downloaded separately

3. **No path slides (19)**:
   - Root cause: **SOURCE DATA MISSING** - MSK slides with no accessible location
   - These were never uploaded to S3 or stored locally
   - Cannot be processed without locating original slide files

### Partial Failures

- **TITAN_SLIDE**: 100% failure rate on all partially-failed slides
- **Other models**: Much lower failure rates (3-55%)
- **Conclusion**: TITAN_SLIDE has a systematic processing issue that needs investigation

---

## Actionable Recommendations

### Priority 1: Missing Slides with S3 Paths (5 slides)
**Impact**: Can recover 5/25 missing slides

Steps:
1. Create new manifest with these 5 slide IDs and S3 paths
2. Stage to Azure Blob Storage using existing staging scripts
3. Submit Azure batch job to process them
4. Expected completion: ~1-2 hours after staging

### Priority 2: Local Path Slide (1 slide)
**Impact**: Can recover 1/25 missing slides

Steps:
1. Create SLURM job for slide 762 (TCGA-PL-A8LX-01)
2. Use local path: `/gpfs/mskmind_ess/limr/repos/mussel-nf/missing_tcga_slides/...`
3. Process with all models
4. Expected completion: ~1 hour

### Priority 3: No Path Slides (19 slides)
**Impact**: Cannot process without source data

Steps:
1. Document these 19 slides as 'SOURCE DATA UNAVAILABLE'
2. Contact MSK data owners to attempt to locate original files
3. Update processing report to indicate these are unrecoverable without source data
4. Consider if these should be excluded from expected total (revise to 43,404 expected)

### Priority 4: TITAN_SLIDE Failures (108 slides)
**Impact**: Can recover complete results for 108 slides

Steps:
1. Investigate TITAN_SLIDE failures in Azure Batch logs
2. Check for common error patterns (OOM, timeout, compatibility issues)
3. Fix the root cause (may need code changes or config adjustments)
4. Create reprocessing manifest for 108 slides with TITAN_SLIDE only
5. Re-run failed slides after fix is confirmed
6. If unfixable: Consider excluding TITAN_SLIDE from processing pipeline

---

## Files Generated

- `detailed_failure_analysis.csv` - Categorized list of all 25 missing slides with action items
- `PAPER_REVISIONS_FAILURE_CATEGORIZATION.md` - This comprehensive report

---

## Summary

**Recoverable**: 6 slides (5 in S3 + 1 local) = 24% of missing slides
**Unrecoverable without source data**: 19 slides = 76% of missing slides
**Needs investigation**: 108 partial failures (TITAN_SLIDE issue)

**Final potential success rate**: 43,296/43,423 = 99.71% (if we recover the 6 slides and fix TITAN_SLIDE)
