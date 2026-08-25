#!/bin/bash
module purge
module load miniforge gcc/11.1.0 cuda/11.1 
module load cudnn/8.6.0_cuda11.x
source activate AMPCliff

export PYTHONUNBUFFERED=1

for bacterium in s_aureus e_coli
do
    for pooling in max mean attn
    do
            python -u downstream_train.py \
                    model.config_dir="/data/public/models/facebook/esm2_t12_35M_UR50D/"\
                    model.regression.version="esm2_t12"\
                    model.regression.pooling=$pooling\
                    model.regression.check_point=false \
                    data.regression.fix.train_file="/data/run01/scv6872/kwli/AMPCliff/data/blast/blosum62 average/diff_5/grampa_${bacterium}_7_25-train.csv" \
                    data.regression.fix.valid_file="/data/run01/scv6872/kwli/AMPCliff/data/blast/blosum62 average/diff_5/grampa_${bacterium}_7_25-valid.csv" \
                    data.regression.fix.test_file="/data/run01/scv6872/kwli/AMPCliff/data/blast/blosum62 average/diff_5/grampa_${bacterium}_7_25-test.csv" \
                    hydra.run.dir="outputs/esm2_t12-${bacterium}-${pooling}-train/${now:%Y-%m-%d}/${now:%H-%M-%S}"\
            
            python -u downstream_train.py \
                    model.config_dir="/data/public/models/facebook/esm2_t33_650M_UR50D/"\
                    model.regression.version="esm2_t33"\
                    model.regression.pooling=$pooling\
                    model.regression.check_point=false \
                    data.regression.fix.train_file="/data/run01/scv6872/kwli/AMPCliff/data/blast/blosum62 average/diff_5/grampa_${bacterium}_7_25-train.csv" \
                    data.regression.fix.valid_file="/data/run01/scv6872/kwli/AMPCliff/data/blast/blosum62 average/diff_5/grampa_${bacterium}_7_25-valid.csv" \
                    data.regression.fix.test_file="/data/run01/scv6872/kwli/AMPCliff/data/blast/blosum62 average/diff_5/grampa_${bacterium}_7_25-test.csv" \
                    hydra.run.dir="outputs/esm2_t33-${bacterium}-${pooling}-train/${now:%Y-%m-%d}/${now:%H-%M-%S}"
    done
done
