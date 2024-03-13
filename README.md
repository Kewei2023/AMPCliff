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
```
sbatch --gpus=1 downstream_train_.sh
```
##### Changing Models

the code is developing, will update a easier version in the future.

***remark for myself**: SEPERATE LLMs with other type of public models*

***to WHOM running this code**: remember `features.type` and `model.regression.version` should be the same value*

**1. modify `configs/downstream.yaml`**
```
features:
  type: gpt2-large 

... # ignore some other settings

model:
  config_dir: "/data/public/models/gpt2-large/" 
  regression:
    version: gpt2-large 
    initial:
      model_name: "/data/public/models/gpt2-large/" 

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

3. modify `factory/initializer.py`
```
if 'gpt2' in self.cfg.model[self.cfg.task.type].version:
                
    config = AutoConfig.from_pretrained(self.cfg.model.config_dir)
    config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.config_dir)
    tokenizer.pad_token = tokenizer.eos_token # in most cases, tokenizer need to be modified
    
    model_base = AutoModel.from_pretrained(self.cfg.model.config_dir).to(self.device)
    
    model = self.PretrainModel(model_base, self.device)
    
    config.problem_type = "regression"
    config.num_labels = 1
    config.hidden_dropout_prob = 0
    train_model = RegModel_v1(model,config).to(self.device)
``` 

2. modify `features/feature_fetcher.py`
```
if 'gpt2' in self.cfg.type or self.cfg.type == 'bert-base': # usually just modify this condition
        
        peptides_spaced = [' '.join(peptide) for peptide in peptides]
        
        if self.task == 'pretrain':
          tokenize_function_partial = partial(self.tokenize_function, add_special_tokens=False)
        else:
          tokenize_function_partial = partial(self.tokenize_function, add_special_tokens=True)
        
        peptides_descriptors = tokenize_function_partial(peptides_spaced)
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