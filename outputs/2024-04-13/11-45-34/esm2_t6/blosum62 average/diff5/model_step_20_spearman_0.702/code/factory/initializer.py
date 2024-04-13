from transformers import AutoModel,AutoTokenizer,AutoConfig,BertConfig, BertModel, BertTokenizer
from tokenizers import Tokenizer
from ..progen2.models.progen.modeling_progen import ProGenModel,ProGenForCausalLM
from ..progen2.models.progen.configuration_progen import ProGenConfig
from ..utils.utils import get_device,fix_random_seed, load_weights, load_model
from .regression import RegModel_v1
from .AMPSpace import LstmNet
from .ML import ModelRegressor
from .peptimizer import Regressor
from .CellTree import CNNRegressor,RNNRegressor
# from .SeqUNet import UNet
# from .generation import CVAEModel
import ipdb


class ModelInitializer():
    def __init__(self,cfg, device):
        
        self.cfg = cfg
        self.device = device
    
    def init(self):

        tokenizer = None
        if self.cfg.task.type == 'regression':
              
            if 'progen' in self.cfg.model[self.cfg.task.type].version:
                config = ProGenConfig.from_pretrained(self.cfg.model.config_dir)
                config.output_hidden_states=True
                config.problem_type = "regression"
                config.num_labels = 1
                config.hidden_dropout_prob = 0
                model = ProGenModel.from_pretrained(self.cfg.model.config_dir, config=config).to(self.device)
                tokenizer = self.create_tokenizer_custom(file=self.cfg.model.tokenizer)
                
                train_model = RegModel_v1(model,config).to(self.device)
                
                
            if 'protbert' in self.cfg.model[self.cfg.task.type].version:
                config = BertConfig.from_pretrained(self.cfg.model.config_dir)
                config.output_hidden_states=True
                config.problem_type = "regression"
                config.num_labels = 1
                config.hidden_dropout_prob = 0
                model = BertModel.from_pretrained(self.cfg.model.config_dir, config=config).to(self.device)
                tokenizer = BertTokenizer.from_pretrained(self.cfg.model.config_dir)
                
                train_model = RegModel_v1(model,config).to(self.device)
                
            if 'esm2' in self.cfg.model[self.cfg.task.type].version:
            
                config = AutoConfig.from_pretrained(self.cfg.model.config_dir)
                config.output_hidden_states = True
                config.problem_type = "regression"
                config.num_labels = 1
                
                tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.config_dir)
                model = AutoModel.from_pretrained(self.cfg.model.config_dir).to(self.device)
                
                
                train_model = RegModel_v1(model,config).to(self.device)
            
            if 'gpt2' in self.cfg.model[self.cfg.task.type].version:
                
                config = AutoConfig.from_pretrained(self.cfg.model.config_dir)
                config.output_hidden_states = True
                config.problem_type = "regression"
                config.num_labels = 1
                config.hidden_dropout_prob = 0
                
                tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.config_dir)
                tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModel.from_pretrained(self.cfg.model.config_dir).to(self.device)
                    
                train_model = RegModel_v1(model,config).to(self.device)
            
            if self.cfg.model[self.cfg.task.type].version == 'bert-base':
                
                config = AutoConfig.from_pretrained(self.cfg.model.config_dir)
                config.output_hidden_states = True
                config.problem_type = "regression"
                config.num_labels = 1
                config.hidden_dropout_prob = 0
                
                tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.config_dir)
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
                model = AutoModel.from_pretrained(self.cfg.model.config_dir).to(self.device)
            
                train_model = RegModel_v1(model,config).to(self.device)
                
            if self.cfg.model[self.cfg.task.type].version == 'AMPSpace':
                
                train_model = LstmNet(embedding_dim=50, 
                                      hidden_num=128, 
                                      num_layer=2, 
                                      bidirectional=False, 
                                      dropout=0.7
                                      ).to(self.device)
            
            if self.cfg.model[self.cfg.task.type].version == 'SeqUNet':
                
                train_model = UNet().to(self.device)
            
            if self.cfg.model[self.cfg.task.type].version == 'peptimizer':
                
                train_model = Regressor(n_filters=256, 
                                        kernel_size=2, 
                                        dropout=0.1, 
                                        input_shape=(self.cfg.data.max_length,2048)
                                      ).to(self.device)

            if self.cfg.model[self.cfg.task.type].version == 'CellFree-rnn':
                
                train_model = RNNRegressor().to(self.device)
            
            if self.cfg.model[self.cfg.task.type].version == 'CellFree-cnn':
                
                train_model = CNNRegressor(self.cfg.data.max_length).to(self.device)
        
        
        return train_model, tokenizer
        
        
    def create_tokenizer_custom(self,file):
      with open(file, 'r') as f:
          return Tokenizer.from_str(f.read())
    