#!/bin/bash
module load anaconda/2020.11 gcc/11.1.0 cuda/11.1 
module load cudnn/8.6.0_cuda11.x
source activate AMPCliff

export PYTHONUNBUFFERED=1
python extract_features.py