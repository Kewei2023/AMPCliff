#!/bin/bash
module load anaconda/2020.11 gcc/11.1.0 # cuda/11.1 gcc/11.1.0
# module load cudnn/8.1.0.77_CUDA11.1
source activate AMPCliff

export PYTHONUNBUFFERED=1
python downstream_evaluate.py