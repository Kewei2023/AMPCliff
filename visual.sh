#!/bin/bash
module load anaconda/2020.11 gcc/11.1.0 cuda/11.1 
module load cudnn/8.6.0_cuda11.x
source activate AMPCliff
# export MLFLOW_TRACKING_URI=http://192.168.1.23:5008
# export MLFLOW_EXPERIMENT_NAME=breeze
# mlflow server --backend-store-uri /data/home/scv6872/Breeze/mlruns --default-artifact-root /data/home/scv6872/Breeze/mlruns --host 0.0.0.0

export PYTHONUNBUFFERED=1
python visual.py