# maintained by kewei li
from transformers import AutoModel,AutoTokenizer,AutoConfig,BertConfig, BertModel, BertTokenizer
from tokenizers import Tokenizer
from ..progen2.models.progen.modeling_progen import ProGenModel,ProGenForCausalLM
from ..progen2.models.progen.configuration_progen import ProGenConfig
from ..utils.utils import get_device,fix_random_seed, load_weights, load_model
from ..utils.std_logger import Logger
from .regression import *  # RegModel_v1, RegModel_v2, RegModel_MLTP
from .pooling import (
    get_supported_poolings,
    resolve_pooling_kwargs,
    validate_pooling_name,
)
from .pooling.llm_pooling_dropin import resolve_mltp_method_kwargs
from .AMPSpace import LstmNet
from .ML import ModelRegressor
from .peptimizer import Regressor
from .CellTree import CNNRegressor,RNNRegressor
# from .SeqUNet import UNet
# from .generation import CVAEModel


def _attach_llm_pooling_kwargs(config, _reg):
    """Attach merged pooling kwargs for ClassificationHead* / build_pooling_modules."""
    config.pooling_kwargs = resolve_pooling_kwargs(_reg)
    if getattr(config, "pooling", None) == "mltp_paper":
        config.mltp_method_kwargs = resolve_mltp_method_kwargs(_reg)


class ModelInitializer():
    def __init__(self,cfg, device):
        
        self.cfg = cfg
        self.device = device
    
    def init(self):

        tokenizer = None
        supported_poolings = get_supported_poolings()
        if self.cfg.task.type == 'regression':
              
            if 'progen' in self.cfg.model[self.cfg.task.type].version:
                config = ProGenConfig.from_pretrained(self.cfg.model.config_dir)
                config.output_hidden_states=True
                config.problem_type = "regression"
                config.num_labels = 1
                config.hidden_dropout_prob = 0
                model = ProGenModel.from_pretrained(self.cfg.model.config_dir, config=config).to(self.device)
                tokenizer = self.create_tokenizer_custom(file=self.cfg.model.tokenizer)

                
                
            if 'protbert' in self.cfg.model[self.cfg.task.type].version:
                config = BertConfig.from_pretrained(self.cfg.model.config_dir)
                config.output_hidden_states=True
                config.problem_type = "regression"
                config.num_labels = 1
                config.hidden_dropout_prob = 0
                model = BertModel.from_pretrained(self.cfg.model.config_dir, config=config).to(self.device)
                tokenizer = BertTokenizer.from_pretrained(self.cfg.model.config_dir)
                

            if 'esm2' in self.cfg.model[self.cfg.task.type].version:
                # Map version to correct config_dir if not explicitly set via command line override
                version = self.cfg.model[self.cfg.task.type].version
                config_dir_mapping = {
                    'esm2_t6': '/data/public/models/facebook/esm2_t6_8M_UR50D/',
                    'esm2_t12': '/data/public/models/facebook/esm2_t12_35M_UR50D/',
                    'esm2_t33': '/data/public/models/facebook/esm2_t33_650M_UR50D/',
                    'esm2_t48': '/data/public/models/facebook/esm2_t48_15B_UR50D/',
                }
                if version in config_dir_mapping:
                    config_dir = config_dir_mapping[version]
                else:
                    config_dir = self.cfg.model.config_dir
                Logger.info(f"Loading ESM2 model: version={version}, path={config_dir}")

                config = AutoConfig.from_pretrained(config_dir)
                config.output_hidden_states = True
                config.problem_type = "regression"
                config.num_labels = 1
                config.output_attentions = True
                config.pooling = self.cfg.model[self.cfg.task.type].pooling
                tokenizer = AutoTokenizer.from_pretrained(config_dir)
                model = AutoModel.from_pretrained(config_dir, config=config).to(self.device)
                

            if 'gpt2' in self.cfg.model[self.cfg.task.type].version:
                
                config = AutoConfig.from_pretrained(self.cfg.model.config_dir)
                config.output_hidden_states = True
                config.problem_type = "regression"
                config.num_labels = 1
                config.hidden_dropout_prob = 0
                
                tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.config_dir)
                tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModel.from_pretrained(self.cfg.model.config_dir).to(self.device)
                

            if self.cfg.model[self.cfg.task.type].version == 'bert-base':
                
                config = AutoConfig.from_pretrained(self.cfg.model.config_dir)
                config.output_hidden_states = True
                config.problem_type = "regression"
                config.num_labels = 1
                config.hidden_dropout_prob = 0
                
                tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.config_dir)
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
                model = AutoModel.from_pretrained(self.cfg.model.config_dir).to(self.device)
            
            
            if self.cfg.features.type == 'LLM':
                raw_pooling = getattr(self.cfg.model[self.cfg.task.type], "pooling", "mean")

                pooling = validate_pooling_name(
                    raw_pooling,
                    allowed=supported_poolings,
                    context="model.regression.pooling",
                )
                config.pooling = pooling
                _reg = self.cfg.model[self.cfg.task.type]
                config.version = self.cfg.model[self.cfg.task.type].version
                _attach_llm_pooling_kwargs(config, _reg)

                if pooling == "mltp_paper":
                    train_model = RegModel_MLTP_Paper(model, config).to(self.device)
                else:
                    train_model = RegModel_v2(model, config).to(self.device)


            if self.cfg.model[self.cfg.task.type].version == 'AMPSpace':
                
                train_model = LstmNet(embedding_dim=50, 
                                      hidden_num=128, 
                                      num_layer=2, 
                                      bidirectional=False, 
                                      dropout=0.7
                                      ).to(self.device)
            
            
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
    