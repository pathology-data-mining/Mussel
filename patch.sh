#!/bin/sh

# activate conda environment
source /home/boehmk/.bashrc
conda activate mussel

# navigate to directory
cd /gpfs/mskmind_ess/boehmk/python_bin/mussel

python tessellate.py --save_dir "/gpfs/mskmind_ess/boehmk/scratch" --slide_file_path "/gpfs/mskmind_emc/data_large/pathology/BR_20-226/slides/${1}.svs" --mpp 1.0 --patch_size 224 --step_size 896
