# Azure Incremental Staging Fix Summary

## Issues Fixed

### Issue 1: Too Few Tasks Created (5 instead of 244)

**Problem**: For 1951 slides with `slides_per_task=8`, only ~5 tasks were created instead of the expected ~244 tasks.

**Root Cause**: The batch submission threshold was set to `batch_size * 5` (40 slides), meaning tasks were only submitted after accumulating 40 slides. If staging was slow or interrupted, most slides never got submitted.

**Fix**: Changed submission threshold from `batch_size * 5` to `batch_size` in line 3234:
```python
# Before:
if len(submission_batch) >= batch_size * 5:  # Submit in larger batches for efficiency
    submit_batch_incrementally()

# After:
if len(submission_batch) >= batch_size:  # Submit every batch_size slides
    submit_batch_incrementally()
```

**Result**: Tasks now submitted every 8 slides, ensuring all ~244 tasks are created as slides are staged.

---

### Issue 2: All Tasks Named `batch_1_of_1_OPTIMUS_plus4more`

**Problem**: All tasks had the same name `batch_1_of_1_OPTIMUS_plus4more` instead of unique names like `batch_1_of_244`, `batch_2_of_244`, etc.

**Root Cause**: The incremental submission function created a temporary CSV with only the current batch (8 slides), then called `submit_tasks_from_csv()` with that CSV. This function calculated `total_batches` based on the temp CSV, always getting `total_batches=1`.

**Fix**: Added batch tracking parameters to maintain global state:

1. Added parameters to `submit_tasks_from_csv()`:
   ```python
   batch_offset: int = 0,  # Starting batch number
   total_batches_global: Optional[int] = None,  # Total batches across all submissions
   ```

2. Calculate total batches upfront in incremental submission:
   ```python
   total_batches_expected = (len(slides) + batch_size - 1) // batch_size  # = 244 for 1951 slides
   current_batch_num = 0  # Track which batch we're on
   ```

3. Pass global context when submitting each batch:
   ```python
   submitter.submit_tasks_from_csv(
       ...
       batch_offset=current_batch_num - 1,  # Zero-indexed offset
       total_batches_global=total_batches_expected,  # 244
   )
   ```

4. Update task ID generation to use global values:
   ```python
   # Before:
   batch_num = batch_idx // slides_per_task + 1
   total_batches = (len(slides) + slides_per_task - 1) // slides_per_task
   
   # After:
   total_batches = total_batches_global if total_batches_global else (...)
   batch_num = batch_offset + (batch_idx // slides_per_task + 1)
   ```

**Result**: Tasks now correctly named:
- `batch_1_of_244_OPTIMUS_plus4more`
- `batch_2_of_244_OPTIMUS_plus4more`
- `batch_3_of_244_OPTIMUS_plus4more`
- ...
- `batch_244_of_244_OPTIMUS_plus4more`

---

## Testing

For job `pr-job-prod-20251204_131857` with 1951 slides:

**Before fixes**:
- Only ~5 tasks created (40 slides * 5 = 200 slides processed)
- All tasks named `batch_1_of_1_OPTIMUS_plus4more`

**After fixes**:
- All ~244 tasks created (1951 slides / 8 slides_per_task)
- Tasks correctly numbered from 1 to 244
- Clear progress tracking during staging

---

## Files Modified

- `scripts/azure_batch/submit_batch_jobs.py`
  - Line 1530-1532: Added `batch_offset` and `total_batches_global` parameters
  - Line 1783: Use `total_batches_global` if provided
  - Line 1794: Calculate `batch_num` with offset for global numbering
  - Line 3083-3084: Calculate and track global batch numbers
  - Line 3164-3165: Increment batch counter in submission loop
  - Line 3202-3203: Pass batch tracking to `submit_tasks_from_csv()`
  - Line 3234: Changed threshold from `batch_size * 5` to `batch_size`

---

## Commits

1. `acce28e` - Fix Azure incremental task naming to show correct batch numbers
   - Includes both the batch naming fix and submission frequency fix
2. Previous: `8a64b5d` - Add SLURM batch size guide (unrelated)
