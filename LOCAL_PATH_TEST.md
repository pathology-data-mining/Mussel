# Local Path Support Test

## Verification

Local paths ARE fully supported in the Azure Batch implementation:

### Input Slides (Local)
```bash
# CSV with local paths
slide_id,slide_path
slide1,/local/path/slide1.svs
slide2,/local/path/slide2.svs
```
**Result**: Files remain local, no staging needed

### Output Paths (Local)
```bash
# Using local OUTPUT_DIR
--output-dir "/mnt/output"
```
**Result**: 
- Creates directory if needed
- Writes results to `/mnt/output/slide1_features.h5`
- Files remain on the compute node

### Code Flow for Local Outputs

1. **Path Preparation** (lines 475-479):
   ```bash
   else
       LOCAL_OUTPUT_H5_PATH="$OUTPUT_H5_PATH"  # No temp path needed
       LOCAL_OUTPUT_PT_PATH="$OUTPUT_PT_PATH"
   ```

2. **Directory Creation** (lines 482-486):
   ```bash
   OUTPUT_DIR=$(dirname "$LOCAL_OUTPUT_H5_PATH")
   if [ ! -d "$OUTPUT_DIR" ]; then
       log "Creating output directory: $OUTPUT_DIR"
       mkdir -p "$OUTPUT_DIR"
   fi
   ```

3. **Processing** (lines 520-750):
   - Writes directly to `LOCAL_OUTPUT_H5_PATH`
   - No upload step

4. **Completion** (lines 795-804):
   ```bash
   else  # Local output
       if [ -f "$MODEL_H5_PATH" ]; then
           log "Output H5 file: $MODEL_H5_PATH (size: ...)"
       fi
   ```

### Usage Example

```bash
# Submit batch job with local outputs
uv run python submit_batch_jobs.py \
  --batch-account-name "ocra" \
  --batch-account-key "$AZURE_BATCH_ACCOUNT_KEY" \
  --batch-account-url "https://ocra.eastus2.batch.azure.com" \
  --storage-account-name "mskpdmgen2" \
  --storage-account-key "$AZURE_STORAGE_KEY" \
  --azure-files-share-name "mussel-staging" \
  --config azure_test.yaml \
  --csv-manifest test_slides.csv \
  --output-dir "/mnt/output" \
  --job-id "mussel-local-test"
```

### Expected Behavior

bœ… **Input**: Local slide files â†’ Staged to Azure Files (accessible from nodes)  
bœ… **Models**: Local model files â†’ Staged to Azure Files (accessible from nodes)  
bœ… **Processing**: Runs on Azure Batch compute nodes  
bœ… **Output**: Written to `/mnt/output/` on compute nodes  

**Note**: Results on compute nodes are NOT automatically retrieved. To access them:
- Use Azure Files for inputs/outputs (recommended)
- Use Azure Blob or S3 for outputs (recommended)
- Or implement custom retrieval mechanism for node-local files

### Recommendation

While local paths are supported during processing, for Azure Batch workflows it's **recommended** to use remote storage for outputs:
- Azure Files: `azfiles://account/share/path`
- Azure Blob: `https://account.blob.core.windows.net/container/path`
- S3: `s3://bucket/path`

This ensures results are accessible after task completion.

