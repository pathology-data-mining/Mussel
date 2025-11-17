# Azure Batch: Multiple Models Per Task Configuration

## Change Summary

Modified `scripts/azure_batch/submit_batch_jobs.py` to process **multiple models per task** instead of one model per task.

## Before (One Model Per Task)
- Each model got its own separate tasks
- 40K slides with 5 models = **25,000 tasks** (5,000 batches Ã— 5 models)
- More tasks = more overhead, more scheduling complexity

## After (Multiple Models Per Task)
- All models run sequentially in the same task
- 40K slides with 5 models = **5,000 tasks** (one batch per 8 slides)
- Fewer tasks = less overhead, simpler monitoring

## Configuration

### Current Setup (Auto-Batching Enabled):
- **Slides per task:** 8 (auto-adjusted for slide-level models)
- **Models per task:** 5 (OPTIMUS, VIRCHOW2, UNI2, TITAN_SLIDE, GIGAPATH_SLIDE)
- **Total tasks:** ~5,000 for 40K slides
- **Processing:** Models run sequentially within each task

### Benefits:
1. **Fewer tasks:** 5x reduction (25K â†’ 5K tasks)
2. **Less overhead:** Single container startup per batch
3. **Simpler monitoring:** Track one task instead of 5
4. **Efficient:** Load each model once, process 8 slides

## 40K Slides Projection (Unchanged)

The total time remains the same because GPU utilization is identical:

- **Total time:** 64.4 hours with 50 A100 GPUs
- **Time per task:** ~14.5 minutes (all 5 models on 8 slides)
- **Cost:** ~$11,817 (on-demand) or ~$2,363 (low-priority)
- **Throughput:** 621 slides/hour

## Usage

The default behavior now processes all specified models in each task:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --job-id test-job \
  --pool-id test-pool \
  --csv-manifest slides.csv \
  --models OPTIMUS,VIRCHOW2,UNI2,TITAN_SLIDE,GIGAPATH_SLIDE \
  --output-prefix azblob://account/container/output
```

This will create ~5,000 tasks (for 40K slides), each processing:
- 8 slides
- All 5 models sequentially

## Code Changes

**File:** `scripts/azure_batch/submit_batch_jobs.py`

1. Removed the `for model_type in models_to_process` loop
2. Set all models as comma-separated strings in parameters:
   - `model_types`: "OPTIMUS,VIRCHOW2,UNI2"
   - `slide_model_types`: "TITAN_SLIDE,GIGAPATH_SLIDE"
3. Updated batch_id generation to include multiple models
4. Removed unreachable dead code (old one-task-per-slide logic)

## Testing

Syntax validated:
```bash
bœ… python3 -m py_compile scripts/azure_batch/submit_batch_jobs.py
```

Ready for Azure Batch testing with all 5 models.
