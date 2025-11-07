#!/bin/bash
# Example: Using CSV manifest with YAML configuration file for batch processing with SLURM

# This example demonstrates how to use CSV manifests with YAML config files
# for better organization and parameter management

set -e

echo "=== Batch Processing with CSV + YAML Config Example ==="
echo

# Create example CSV manifest
cat > slides_manifest.csv << 'EOF'
slide_id,slide_path
slide_001,s3://my-bucket/slides/slide_001.svs
slide_002,s3://my-bucket/slides/slide_002.svs
slide_003,s3://my-bucket/slides/slide_003.svs
EOF

echo "Created slides_manifest.csv"
echo

# Create example YAML config file with parameters
cat > batch_params.yaml << 'EOF'
# Processing parameters
prefilter_model_type: CTRANSPATH
batch_size: 64
num_workers: 4
use_gpu: true
segment_threshold: 0
patch_size: 256
mpp: 0.5
EOF

echo "Created batch_params.yaml"
echo

# Submit to SLURM using CSV manifest and config file (dry run - remove --submit flag)
echo "Submitting to SLURM (dry run)..."

# Check if AWS credentials are set
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
  echo "WARNING: AWS credentials not set. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
  echo "Proceeding with dry run only..."
fi

python3 scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides_manifest.csv \
  --config batch_params.yaml \
  --output-s3-prefix s3://my-bucket/results \
  --partition gpu \
  --gres gpu:1 \
  --cpus-per-task 8 \
  --mem 32G \
  --time 04:00:00 \
  ${AWS_ACCESS_KEY_ID:+--aws-access-key-id "$AWS_ACCESS_KEY_ID"} \
  ${AWS_SECRET_ACCESS_KEY:+--aws-secret-access-key "$AWS_SECRET_ACCESS_KEY"}
  # Add --submit flag to actually submit jobs

echo
echo "=== Dry run complete ==="
echo "Generated SLURM batch scripts or job array"
echo
echo "To actually submit:"
echo "  Add --submit flag to the command above"
echo
echo "Advantages of CSV + config approach:"
echo "  ✓ Simple CSV with just slide IDs and paths"
echo "  ✓ All processing parameters in config file"
echo "  ✓ Easy to update parameters without touching slide list"
echo "  ✓ Configuration tracked in result manifests"
echo "  ✓ Human-readable YAML format with comments"

