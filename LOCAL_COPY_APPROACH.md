# Local Copy Approach for Remote Storage

## Date
2025-11-17 06:55 UTC

## Problem

Tasks were failing when trying to read HDF5 files directly from Azure Blob Storage for two-step aggregation:

```
FileNotFoundError: Unable to open file 'az://mussel-results/...features.h5'
```

While h5py/fsspec can *write* to remote storage, *reading* HDF5 files from remote storage in aggregation workflows has limitations.

## Solution

**Changed approach from "write directly to remote" to "write local, then copy":**

1. ‚úÖ Process files to **local temporary directory**
2. ‚úÖ After processing succeeds, **copy all files to remote storage**
3. ‚úÖ **Clean up** local temporary directory

This ensures:
- All file operations (read/write) are local and fast
- No issues with h5py/fsspec remote file access
- Files are uploaded as a batch at the end
- Cleanup happens automatically

## Implementation

### Modified File
`scripts/azure_batch/run_tessellate_extract_features.sh`

### Changes

**1. Detect Remote Output (lines 238-256)**
```bash
# Check if output is remote (azblob://, az://, s3://, etc.)
if [[ "$REMOTE_OUTPUT_DIR" =~ ^(azblob://|az://|s3://|http://|https://) ]]; then
    log "Remote output detected: $REMOTE_OUTPUT_DIR"
    log "Using local temp directory for processing, will upload at end"
    IS_REMOTE_OUTPUT=true
    LOCAL_OUTPUT_DIR="${WORK_DIR}/output"
    mkdir -p "$LOCAL_OUTPUT_DIR"
    EFFECTIVE_OUTPUT_DIR="$LOCAL_OUTPUT_DIR"
else
    EFFECTIVE_OUTPUT_DIR="$REMOTE_OUTPUT_DIR"
fi
```

**2. Upload After Success (lines 415-498)**
```bash
# Upload output files to remote storage if needed
if [ "$IS_REMOTE_OUTPUT" = true ]; then
    log "Uploading output files to remote storage: $REMOTE_OUTPUT_DIR"
    
    # Azure Blob: Use azcopy or az storage blob upload-batch
    # S3: Use aws s3 sync
    
    if [ $UPLOAD_EXIT -eq 0 ]; then
        log "SUCCESS: Files uploaded"
        rm -rf "$LOCAL_OUTPUT_DIR"  # Cleanup
    fi
fi
```

## Upload Methods Supported

### Azure Blob Storage
- **Primary**: `azcopy copy` (fast, batch upload)
- **Fallback**: `az storage blob upload-batch`
- Requires: `AZURE_STORAGE_ACCOUNT` and `AZURE_STORAGE_KEY`

### S3
- **Method**: `aws s3 sync`
- Requires: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

### Local
- **Method**: Direct write (no upload needed)

## Benefits

búÖ **Reliability**: All file I/O is local - no network issues during processing
búÖ **Performance**: Local disk I/O is much faster than remote
búÖ **Compatibility**: No h5py/fsspec remote reading issues
búÖ **Atomic**: Files only appear in remote storage after successful completion
búÖ **Clean**: Automatic cleanup of temporary files

## Test Plan

1. Process 2 slides with OPTIMUS (patch-level model)
2. Process 2 slides with TITAN_SLIDE (slide-level model with aggregation)
3. Verify:
   - Files written locally during processing
   - Files uploaded to Azure Blob after success
   - Local temp directory cleaned up
   - Both tasks complete successfully

## Expected Workflow

```
1. Download slides from Azure Blob      ‚úÖ (already working)
2. Tessellate slides                    ‚úÖ (already working)
3. Extract features ‚Üí local dir         ‚úÖ (NEW: write local)
4. Aggregate features (read local)      ‚úÖ (NEW: read local files)
5. Upload results to Azure Blob         üîÑ (NEW: batch upload)
6. Clean up local files                 üîÑ (NEW: cleanup)
```

## Files Modified

- `scripts/azure_batch/run_tessellate_extract_features.sh` (lines 238-256, 415-498)

## Previous Approaches (Abandoned)

bùå **Direct remote write with fsspec**: Failed on reading h5 files back
bùå **Two separate output paths**: Too complex, error-prone

búÖ **Local write + batch copy**: Simple, reliable, performant

