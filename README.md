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

#### Deep Learning Method

**single GPU**

*NOTE:* set `ddp` in `downstream.yaml` to `false`.
```
sbatch --gpus=1 downstream_train.sh
```

**multi GPUs**


*NOTE:* 
1. set `ddp` in `downstream.yaml` to `true`.

2. `CUDA_VISIBLE_DEVICES` start from 0, and `WORLD_SIZE`=`--nproc_per_node`=len(`CUDA_VISIBLE_DEVICES`)=`--gpus`!

```
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
```

```
sbatch --gpus=3 distribute_train.sh # maximum 8
```
##### Changing Models

the code is developing, will update a easier version in the future.

***remark for myself**: SEPERATE LLMs with other type of public models*

***to WHOM running this code**: for LLM, `features.type` always `LLM`*

**1. modify `configs/downstream.yaml`**
```
model:
  config_dir: "/data/public/models/gpt2-large/" 
  regression:
    version: gpt2-large 

``` 

Here we support the following models:
| Models      | config_dir                           |model_name|
|---------------|---------------------------------------|-|
| bert-base     | `/data/public/models/bert-base-uncased/`|`/data/public/models/bert-base-uncased/`|
| protgpt2     | `/data/public/models/ProtGPT2/`|`/data/public/models/ProtGPT2/`|
| gpt2-large     | `/data/public/models/gpt2-large/`|`/data/public/models/gpt2-large/` |
| esm2_t12     | `/data/public/models/facebook/esm2_t12_35M_UR50D/`|`/data/public/models/facebook/esm2_t12_35M_UR50D/` |
| remained to be fullfilled    |remained to be fullfilled|remained to be fullfilled|
| CellTree-cnn     | -|-|
| CellTree-rnn     | -|-|
| AMPSpace     | -|-|
| SeqUNet     | -|-|
| peptimizer     | -|-|

For more LLM models, please see `/data/public/models/`, feel free to play with it!

2. modify `factory/initializer.py`
```
if 'gpt2' in self.cfg.model[self.cfg.task.type].version:
                
    config = AutoConfig.from_pretrained(self.cfg.model.config_dir)
    config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.config_dir)
    tokenizer.pad_token = tokenizer.eos_token # in most cases, tokenizer need to be modified
    
    model = AutoModel.from_pretrained(self.cfg.model.config_dir).to(self.device)
    
    
    config.problem_type = "regression"
    config.num_labels = 1
    config.hidden_dropout_prob = 0
    train_model = RegModel_v1(model,config).to(self.device)
``` 

#### Machine Learning Method
```
sbatch --gpus=1 machine_learning_train.sh
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