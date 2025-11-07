#!/bin/bash
# Example: Using YAML configuration file for batch processing with SLURM

# This example demonstrates how to use YAML config files instead of CSV manifests
# for better organization and default parameter management

set -e

echo "=== Batch Processing with YAML Config Example ==="
echo

# Create example YAML config file
cat > batch_config_example.yaml << 'EOF'
# Default parameters for all tasks
defaults:
  prefilter_model_type: CTRANSPATH
  batch_size: 64
  num_workers: 4
  use_gpu: true
  segment_threshold: 0
  patch_size: 256
  mpp: 0.5

# Task definitions
tasks:
  - task_id: slide_001
    slide_path: s3://my-bucket/slides/slide_001.svs
    output_h5_path: s3://my-bucket/results/CTRANSPATH/h5/slide_001_features.h5
    output_pt_path: s3://my-bucket/results/CTRANSPATH/pt/slide_001_features.pt
    
  - task_id: slide_002
    slide_path: s3://my-bucket/slides/slide_002.svs
    output_h5_path: s3://my-bucket/results/CTRANSPATH/h5/slide_002_features.h5
    output_pt_path: s3://my-bucket/results/CTRANSPATH/pt/slide_002_features.pt
    # Override batch size for this specific task
    batch_size: 128
EOF

echo "Created batch_config_example.yaml"
echo

# Submit to SLURM using config file (dry run - remove --submit flag)
echo "Submitting to SLURM (dry run)..."

# Check if AWS credentials are set
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
  echo "WARNING: AWS credentials not set. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
  echo "Proceeding with dry run only..."
fi

python3 scripts/slurm/submit_slurm_jobs.py \
  --task-config batch_config_example.yaml \
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
echo "Generated SLURM batch scripts for each task"
echo
echo "To actually submit:"
echo "  Add --submit flag to the command above"
echo
echo "Advantages of config files over CSV:"
echo "  ✓ Define default parameters once"
echo "  ✓ Override parameters per task"
echo "  ✓ Human-readable YAML or JSON format"
echo "  ✓ Configuration tracked in result manifests"
echo "  ✓ Comments supported (YAML)"
