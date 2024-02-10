#!/bin/sh

# activate conda environment
source /home/boehmk/.bashrc
conda activate mussel

# navigate to directory
cd /gpfs/mskmind_ess/boehmk/python_bin/mussel

python extract_features.py \
--model quilt \
--save_dir "/gpfs/mskmind_ess/boehmk/scratch" \
--slide_file_path "/gpfs/mskmind_emc/data_large/pathology/BR_20-226/slides/${1}.svs" \
--patch_file_path "/gpfs/mskmind_ess/boehmk/scratch/patches/${1}.h5" \
--gpus 0 1 2 3