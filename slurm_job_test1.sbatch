#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/test1_%j.out
#SBATCH --error=slurm_logs/test1_%j.err

# Environment setup
export SLIDE_PATH=/test/slide1.svs
export OUTPUT_H5_PATH=/output/slide1.h5
export OUTPUT_PT_PATH=/output/slide1.pt
export CLASSIFIER_THRESHOLD=0.75
export PREFILTER_MODEL_TYPE=CTRANSPATH
export SEGMENT_THRESHOLD=0
export PATCH_SIZE=256
export MPP=0.5
export NUM_WORKERS=4
export BATCH_SIZE=64
export USE_GPU=true
export AWS_DEFAULT_REGION=us-east-1

# Load modules (customize for your environment)
# module load python/3.9
# module load cuda/11.8

# Run task script
bash /home/runner/work/Mussel/Mussel/scripts/common/run_tessellate_extract_features.sh
