#!/bin/bash
# Example: Using CSV manifest with YAML parameters file

# This example demonstrates how to combine:
# - CSV manifest for slide IDs and paths
# - YAML config for processing parameters

set -e

echo "=== CSV + YAML Parameters Example ==="
echo

# Create example CSV manifest with slide paths
cat > slides_manifest.csv << 'EOF'
slide_id,slide_path
slide_001,s3://my-bucket/slides/slide_001.svs
slide_002,s3://my-bucket/slides/slide_002.svs
slide_003,s3://my-bucket/slides/slide_003.svs
EOF

echo "Created slides_manifest.csv"
cat slides_manifest.csv
echo

# Create example YAML config with parameters
cat > params.yaml << 'EOF'
# Processing parameters
prefilter_model_type: CTRANSPATH
batch_size: 64
num_workers: 4
use_gpu: true
patch_size: 256
mpp: 0.5

# Aggregation
aggregation_method: identity
segment_threshold: 0
EOF

echo "Created params.yaml"
cat params.yaml
echo

echo "=== Submission Examples ==="
echo

echo "1. SLURM with CSV + YAML params:"
echo "   python scripts/slurm/submit_slurm_jobs.py \\"
echo "     --csv-manifest slides_manifest.csv \\"
echo "     --config-file-params params.yaml \\"
echo "     --output-s3-prefix s3://bucket/results \\"
echo "     --partition gpu \\"
echo "     --gres gpu:1 \\"
echo "     --submit"
echo

echo "2. HTCondor with CSV + YAML params:"
echo "   python scripts/condor/submit_condor_jobs.py \\"
echo "     --csv-manifest slides_manifest.csv \\"
echo "     --config-file-params params.yaml \\"
echo "     --output-s3-prefix s3://bucket/results \\"
echo "     --submit"
echo

echo "3. Azure Batch with CSV + YAML params:"
echo "   python scripts/azure_batch/submit_batch_jobs.py \\"
echo "     --csv-manifest slides_manifest.csv \\"
echo "     --config-file params.yaml \\"
echo "     --output-s3-prefix s3://bucket/results \\"
echo "     --pool-id my-pool \\"
echo "     --job-id my-job \\"
echo "     --create-pool \\"
echo "     --create-job"
echo

echo "=== Benefits of CSV + YAML Approach ==="
echo "  ✓ Separate slide list from processing parameters"
echo "  ✓ Easy to maintain slide manifest (just ID and path)"
echo "  ✓ All parameters in one readable config file"
echo "  ✓ Parameters tracked in result manifests"
echo "  ✓ Command-line args override config file values"
