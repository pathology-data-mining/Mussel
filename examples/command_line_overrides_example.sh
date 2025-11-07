#!/bin/bash
# Example demonstrating command-line parameter overrides with config files
#
# This script shows how command-line arguments override configuration file parameters.
# Priority order: config file < command-line args < task-specific config

set -e

echo "=========================================="
echo "Command-Line Parameter Overrides Example"
echo "=========================================="
echo ""

# Create a test config file
cat > /tmp/test_params.yaml << 'EOF'
# Baseline processing parameters
prefilter_model_type: CTRANSPATH
batch_size: 64
num_workers: 4
use_gpu: true

# Baseline resource requirements
resources:
  cpus: 8
  memory: 32G
  gpus: 1

# Baseline SLURM settings
slurm:
  partition: gpu
  time: "04:00:00"
EOF

echo "Created baseline config file: /tmp/test_params.yaml"
echo "Contents:"
cat /tmp/test_params.yaml
echo ""

# Create a simple CSV manifest
cat > /tmp/test_slides.csv << 'EOF'
slide_id,slide_path
slide_001,s3://my-bucket/slides/slide_001.svs
slide_002,s3://my-bucket/slides/slide_002.svs
EOF

echo "Created CSV manifest: /tmp/test_slides.csv"
echo "Contents:"
cat /tmp/test_slides.csv
echo ""

echo "=========================================="
echo "Example 1: Baseline (no overrides)"
echo "=========================================="
echo "Command:"
echo "  python scripts/slurm/submit_slurm_jobs.py \\"
echo "    --csv-manifest /tmp/test_slides.csv \\"
echo "    --config /tmp/test_params.yaml"
echo ""
echo "Result: Uses all parameters from config file"
echo "  - batch_size: 64 (from config)"
echo "  - partition: gpu (from config)"
echo "  - cpus_per_task: 8 (from config resources)"
echo "  - mem: 32G (from config resources)"
echo ""

echo "=========================================="
echo "Example 2: Override batch size and partition"
echo "=========================================="
echo "Command:"
echo "  python scripts/slurm/submit_slurm_jobs.py \\"
echo "    --csv-manifest /tmp/test_slides.csv \\"
echo "    --config /tmp/test_params.yaml \\"
echo "    --batch-size 128 \\"
echo "    --partition cpu"
echo ""
echo "Result: Command-line args override config"
echo "  - batch_size: 128 (OVERRIDDEN from command-line, was 64 in config)"
echo "  - partition: cpu (OVERRIDDEN from command-line, was gpu in config)"
echo "  - cpus_per_task: 8 (from config, not overridden)"
echo "  - mem: 32G (from config, not overridden)"
echo ""

echo "=========================================="
echo "Example 3: Override resource requirements"
echo "=========================================="
echo "Command:"
echo "  python scripts/slurm/submit_slurm_jobs.py \\"
echo "    --csv-manifest /tmp/test_slides.csv \\"
echo "    --config /tmp/test_params.yaml \\"
echo "    --cpus-per-task 16 \\"
echo "    --mem 64G \\"
echo "    --time 08:00:00"
echo ""
echo "Result: Resource overrides for larger job"
echo "  - cpus_per_task: 16 (OVERRIDDEN from command-line, was 8 in config)"
echo "  - mem: 64G (OVERRIDDEN from command-line, was 32G in config)"
echo "  - time: 08:00:00 (OVERRIDDEN from command-line, was 04:00:00 in config)"
echo "  - batch_size: 64 (from config, not overridden)"
echo ""

echo "=========================================="
echo "Example 4: Override model type"
echo "=========================================="
echo "Command:"
echo "  python scripts/slurm/submit_slurm_jobs.py \\"
echo "    --csv-manifest /tmp/test_slides.csv \\"
echo "    --config /tmp/test_params.yaml \\"
echo "    --prefilter-model-type UNI"
echo ""
echo "Result: Test different feature extractor"
echo "  - prefilter_model_type: UNI (OVERRIDDEN from command-line, was CTRANSPATH in config)"
echo "  - All other parameters from config unchanged"
echo ""

echo "=========================================="
echo "Example 5: Add parameters not in config"
echo "=========================================="
echo "Command:"
echo "  python scripts/slurm/submit_slurm_jobs.py \\"
echo "    --csv-manifest /tmp/test_slides.csv \\"
echo "    --config /tmp/test_params.yaml \\"
echo "    --postfilter-models VIRCHOW,H_OPTIMUS_0 \\"
echo "    --aggregation-method mean"
echo ""
echo "Result: Add new parameters to config baseline"
echo "  - postfilter_models: VIRCHOW,H_OPTIMUS_0 (NEW from command-line)"
echo "  - aggregation_method: mean (NEW from command-line)"
echo "  - All parameters from config still applied"
echo ""

echo "=========================================="
echo "Example 6: Multiple overrides"
echo "=========================================="
echo "Command:"
echo "  python scripts/slurm/submit_slurm_jobs.py \\"
echo "    --csv-manifest /tmp/test_slides.csv \\"
echo "    --config /tmp/test_params.yaml \\"
echo "    --batch-size 128 \\"
echo "    --prefilter-model-type UNI \\"
echo "    --partition cpu \\"
echo "    --cpus-per-task 16 \\"
echo "    --mem 64G"
echo ""
echo "Result: Multiple simultaneous overrides"
echo "  - batch_size: 128 (OVERRIDDEN)"
echo "  - prefilter_model_type: UNI (OVERRIDDEN)"
echo "  - partition: cpu (OVERRIDDEN)"
echo "  - cpus_per_task: 16 (OVERRIDDEN)"
echo "  - mem: 64G (OVERRIDDEN)"
echo "  - All other config parameters still applied"
echo ""

echo "=========================================="
echo "Key Takeaways"
echo "=========================================="
echo ""
echo "1. Config files provide baseline parameters"
echo "2. Command-line args ALWAYS override config file values"
echo "3. You can override any parameter: processing, resources, backend-specific"
echo "4. You can add new parameters not in the config file"
echo "5. Unspecified parameters still use config file values"
echo ""
echo "Priority order (lowest to highest):"
echo "  1. Config file defaults"
echo "  2. Command-line arguments (override config)"
echo "  3. Task-specific config (in standalone mode only)"
echo ""
echo "This design allows:"
echo "  - Baseline configs for common workflows"
echo "  - Quick parameter adjustments without editing files"
echo "  - Testing parameter variations efficiently"
echo "  - Environment-specific overrides (dev vs prod)"
echo ""

# Cleanup
rm -f /tmp/test_params.yaml /tmp/test_slides.csv

echo "Done! Test files cleaned up."
