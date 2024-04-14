# AMPCliff

testing plenty of models for AMP activity cliff prediction

## Getting Started
### Installation
```
git clone git@github.com:Kewei2023/AMPCliff.git

cd AMPCliff # the folder name must be AMPCliff 
```
### Dependencies

```
conda env create -f environment.yaml
```
### Get Data
```
private_key is needed, contact kewei
```
### Runing

#### Machine Learning Method
```
sbatch --gpus=1 machine_learning_train.sh
```
*for deep learning method, change `features.type` as follows*
Here we support the following models:
| Models      | type|
|---------------|---------------------------------------|
| CellFree-cnn     | CellFree-cnn|
| CellFree-rnn     | CellFree-rnn|
| AMPSpace     | AMPSpace |
| peptimizer     | peptimizer |
#### Deep Learning Method

#### LMs: GLMs and MLMs

**single GPU**

*NOTE:* set `ddp` in `downstream.yaml` to `false`.
```
sbatch --gpus=1 downstream_train.sh
```

**Changing Models**

the code is developing, will update a easier version in the future.

***remark for myself**: SEPERATE LLMs with other type of public models*

*for LLM, `features.type` always `LLM`*

**1. modify `configs/downstream.yaml`**
```
model:
  config_dir: "/data/public/models/gpt2-large/" 
  regression:
    version: gpt2-large 

``` 

Here we support the following models:
| Models      | config_dir                           |
|---------------|---------------------------------------|
| bert-base     | `/data/public/models/bert-base-uncased/`|
| esm2_t6   | `/data/public/models/facebook/esm2_t6_8M_UR50D/`|
| esm2_t12     | `/data/public/models/facebook/esm2_t12_35M_UR50D/`|
| esm2_t33     | `/data/public/models/facebook/esm2_t33_650M_UR50D/`|
| protgpt2     | `/data/public/models/ProtGPT2/`|
| gpt2-base   | `/data/public/models/gpt2/`|
| progen2-small     | `/data/public/models/progen2/progen2_small/`|
| progen2-base     | `/data/public/models/progen2/progen2_base/`|
| progen2-medium     | `/data/public/models/progen2/progen2_medium/`|

**load checkpoints**

set `check_point.load`=`true` and give a model path to `check_point.path`

```
sbatch --gpus=1 downstream_evaluate.sh
```




### MLFlow Setting

**URL:** the URL depends on the IP adress of the machine, for Linux please use `ifconfig` command to check

**Port:** don't change 
```
source activate AMPCliff
conda env config vars set MLFLOW_EXPERIMENT_NAME=breeze
conda env config vars set MLFLOW_S3_ENDPOINT_URL=http://192.168.1.23:5002 #
conda env config vars set MLFLOW_TRACKING_URI=http://192.168.1.23:5002
conda env config vars set REGISTERED_MODEL_NAME=breezeModel
```