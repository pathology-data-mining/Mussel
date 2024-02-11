#!/bin/sh

# activate conda environment
source /home/boehmk/.bashrc
conda activate mussel

# navigate to mussel dir
cd /gpfs/mskmind_ess/boehmk/mussel

# run script with all args
python main.py "$@"
