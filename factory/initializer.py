from ..models.breeze import BreezeModel,BreezeTokenizer,BreezeForMaskedLM,BreezeForSequenceClassification, BreezeConfig
from transformers import AutoModel,AutoTokenizer,AutoConfig,EsmModel,EsmForSequenceClassification,LlamaForCausalLM, LlamaTokenizer, EsmForMaskedLM
from ..utils.utils import get_device,fix_random_seed, load_weights, load_model
from .rank import RankModel
from .regression import RegModel_v2,RegModel_v1
from .AMPSpace import LstmNet
from .ML import ModelRegressor
from .peptimizer import Regressor
from .CellTree import CNNRegressor,RNNRegressor
from .SeqUNet import UNet
from .generation import CVAEModel
import ipdb


class ModelInitializer():
    def __init__(self,cfg, device):
        
        self.cfg = cfg
        self.device = device
    
    def init(self):
        
        
        
        tokenizer = None
        if self.cfg.task.type == 'regression':
              
            if self.cfg.model[self.cfg.task.type].version == 'breeze':
            
                config = BreezeConfig.from_pretrained(self.cfg.model.config_dir)
                config.output_hidden_states = True
                config.problem_type = "regression"
                config.num_labels = 1
                
                tokenizer = BreezeTokenizer.from_pretrained(self.cfg.model.config_dir)
                model_base = BreezeModel(config=config).to(self.device)
                
                model = self.PretrainModel(model_base, self.device)
                
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

            if self.cfg.model[self.cfg.task.type].version == 'CellTree-rnn':
                
                train_model = RNNRegressor().to(self.device)
            
            if self.cfg.model[self.cfg.task.type].version == 'CellTree-cnn':
                
                train_model = CNNRegressor(self.cfg.data.max_length).to(self.device)
        
        if self.cfg.task.type == 'rank':
        
            config = BreezeConfig.from_pretrained(self.cfg.model.config_dir)
            config.output_hidden_states = True
            tokenizer = BreezeTokenizer.from_pretrained(self.cfg.model.config_dir)
            model_base = BreezeModel(config=config).to(self.device)
            
            model = self.PretrainModel(model_base, self.device)
            config.problem_type = "regression"
            config.num_labels = 1
            train_model = RankModel(model,config).to(self.device)

        if self.cfg.task.type == 'generation':
            
            train_model = CVAEModel(encoder = model, 
                                    seq_length = self.cfg.data.max_length,
                                hidden_dim = config.hidden_size, 
                                latent_dim = self.cfg.model.generation.cvae.latent_dim,
                                num_layers = self.cfg.model.generation.cvae.num_layers).to(self.device)
        
        return train_model, tokenizer

    def PretrainModel(self,model_base, device):
        
        if self.cfg.model[self.cfg.task.type].initial.initial_weight:
        
            init_model = EsmModel.from_pretrained(self.cfg.model[self.cfg.task.type].initial.model_name)
            
            init_model.half()
            init_model.to(device)
            # ipdb.set_trace()

            
            pretrained_dict = init_model.state_dict()
            model_dict = model_base.state_dict()
            updated_dict = {name.replace('esm', 'Breeze'): param for name, param in pretrained_dict.items() if 'classifier' not in name}
          
            model_dict.update(updated_dict)
            model_base.load_state_dict(model_dict)

            model = model_base
        
        if self.cfg.model[self.cfg.task.type].check_point.load:
            model = load_model(model_base, self.cfg.model[self.cfg.task.type].check_point.path, device)    

        if len(self.cfg.train.freeze) != 0:
            for i, layer in enumerate(model.encoder.layer):
            # for i, layer in enumerate(model_base.Breeze.encoder.layer):
                if i in self.cfg.train.freeze: 
                    for param in layer.parameters():
                        param.requires_grad = False

            # 检查第一层是否被正确冻结
            for param in model.encoder.layer[0].parameters():
                assert param.requires_grad == False, "Layer not frozen correctly."

        return model