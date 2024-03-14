#!/bin/bash
module load anaconda/2020.11 gcc/11.1.0 # cuda/11.1 gcc/11.1.0
# module load cudnn/8.1.0.77_CUDA11.1
source activate AMPCliff

for i in {0..0}
do
    WORLD_SIZE=3 CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
                --nproc_per_node=3 \
                --nnodes=1          \
                --node_rank=0       \
                --master_addr=localhost  \
                --master_port=22226 \
                downstream_train.py "train.random_seed=$i"
done