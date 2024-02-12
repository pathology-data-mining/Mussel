#!/bin/bash

source ~/.bashrc
conda activate mussel

cd /gpfs/mskmind_ess/boehmk/mussel
python scuba/app.py "cuda:0"
