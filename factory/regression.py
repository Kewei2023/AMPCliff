from typing import Optional
import copy

import torch
import torch.nn as nn
from ..utils.std_logger import Logger
from .pooling import (
    MultiHeadVectorAttnPooling,
    apply_pooling,
    build_pooling_modules,
    validate_pooling_name,
)

from AMPCliff.factory.pooling.llm_pooling_dropin import MultiLayerTrainablePooling

def masked_mean_pooling(features, attention_mask):
    if attention_mask is None:
        return features.mean(1)
    mask = attention_mask.unsqueeze(-1).expand(features.size()).float()
    return (features * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def masked_max_pooling(features, attention_mask):
    if attention_mask is None:
        return features.max(1).values
    mask = attention_mask.unsqueeze(-1).expand(features.size()).float()
    masked_features = features.masked_fill(mask == 0, -1e9)
    return masked_features.max(1).values
def _pool_all_layers(
    all_layer_features: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    head: nn.Module,
) -> torch.Tensor:
    """Pool each layer [B,T,D] with head's pooling -> [B,D,L].
    Caller must pass DC-transformed features when using ConcatDC/DistillVC so dimension matches head.
    """
    B, T, D, L = all_layer_features.shape
    pooling = getattr(head, "pooling", "mean")
    attn_pool = getattr(head, "attn_pool", None)
    sap_pool = getattr(head, "sap_pool", None)
    pooled = []
    for l in range(L):
        feats = all_layer_features[:, :, :, l]
        p = apply_pooling(
            feats,
            attention_mask,
            pooling,
            mean_pooling=masked_mean_pooling,
            max_pooling=masked_max_pooling,
            attn_pool=attn_pool,
            sap_pool=sap_pool,
        )
        pooled.append(p)
    return torch.stack(pooled, dim=-1)


class ClassificationHead3(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.pooling = validate_pooling_name(config.pooling, context="ClassificationHead3 pooling")
        Logger.info(f"pooling: {self.pooling}")
        _pool_kw = getattr(config, "pooling_kwargs", None)
        if _pool_kw is None:
            from .pooling import resolve_pooling_kwargs

            _pool_kw = resolve_pooling_kwargs(config)
        self.attn_pool, self.sap_pool = build_pooling_modules(
            self.pooling,
            d_model=config.hidden_size * 2,
            **_pool_kw,
            attn_factory=lambda d_model: MultiHeadVectorAttnPooling(
                d_model=d_model,
                num_heads=8,
                temperature=1.0,
                gated=True,
                dropout=0.1,
            ),
        )
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features,attention_mask, **kwargs):
        x = apply_pooling(
            features,
            attention_mask,
            self.pooling,
            mean_pooling=masked_mean_pooling,
            max_pooling=masked_max_pooling,
            attn_pool=self.attn_pool,
            sap_pool=self.sap_pool,
        )
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def forward_return_pooled(self, features, attention_mask, **kwargs):
        """Return pooled token representation (before dense / regression head)."""
        return apply_pooling(
            features,
            attention_mask,
            self.pooling,
            mean_pooling=masked_mean_pooling,
            max_pooling=masked_max_pooling,
            attn_pool=self.attn_pool,
            sap_pool=self.sap_pool,
        )

class ClassificationHead2(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.pooling = validate_pooling_name(config.pooling, context="ClassificationHead2 pooling")
        Logger.info(f"pooling: {self.pooling}")
        _pool_kw = getattr(config, "pooling_kwargs", None)
        if _pool_kw is None:
            from .pooling import resolve_pooling_kwargs

            _pool_kw = resolve_pooling_kwargs(config)
        self.attn_pool, self.sap_pool = build_pooling_modules(
            self.pooling,
            d_model=config.hidden_size,
            **_pool_kw,
            attn_factory=lambda d_model: MultiHeadVectorAttnPooling(
                d_model=d_model,
                num_heads=8,
                temperature=1.0,
                gated=True,
                dropout=0.1,
            ),
        )

    def forward(self, features,attention_mask, **kwargs):
        x = apply_pooling(
            features,
            attention_mask,
            self.pooling,
            mean_pooling=masked_mean_pooling,
            max_pooling=masked_max_pooling,
            attn_pool=self.attn_pool,
            sap_pool=self.sap_pool,
        )
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def forward_return_pooled(self, features, attention_mask, **kwargs):
        """Return pooled token representation (before dense / regression head)."""
        return apply_pooling(
            features,
            attention_mask,
            self.pooling,
            mean_pooling=masked_mean_pooling,
            max_pooling=masked_max_pooling,
            attn_pool=self.attn_pool,
            sap_pool=self.sap_pool,
        )

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.pooling = validate_pooling_name(config.pooling, context="ClassificationHead pooling")
        _pool_kw = getattr(config, "pooling_kwargs", None)
        if _pool_kw is None:
            from .pooling import resolve_pooling_kwargs

            _pool_kw = resolve_pooling_kwargs(config)
        self.attn_pool, self.sap_pool = build_pooling_modules(
            self.pooling,
            d_model=config.hidden_size,
            **_pool_kw,
            attn_factory=lambda d_model: MultiHeadVectorAttnPooling(
                d_model=d_model,
                num_heads=6,
                temperature=1.0,
                gated=True,
                dropout=0.1,
            ),
        )
    def forward(self, features, attention_mask=None, **kwargs):
        x = apply_pooling(
            features,
            attention_mask,
            self.pooling,
            mean_pooling=masked_mean_pooling,
            max_pooling=masked_max_pooling,
            attn_pool=self.attn_pool,
            sap_pool=self.sap_pool,
        )
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def forward_return_pooled(self, features, attention_mask=None, **kwargs):
        """Return pooled token representation (before dense / regression head)."""
        return apply_pooling(
            features,
            attention_mask,
            self.pooling,
            mean_pooling=masked_mean_pooling,
            max_pooling=masked_max_pooling,
            attn_pool=self.attn_pool,
            sap_pool=self.sap_pool,
        )

class RegModel_v1(nn.Module):
    '''
    remain modification
    '''
    def __init__(self, pretrain_model,config): # cfg.model.rank.residual
        super(RegModel_v1, self).__init__()
        self.pretrain_model = pretrain_model
        # Assume esm_model outputs features of size N
        # Regression task layers
        
        self.residualcnn_reg =ClassificationHead(config)
        
        # self.regression1 = FeedForward(hidden_dim,hidden_dim,1)
        # self.regression2 = FeedForward(hidden_dim,hidden_dim,1)
        # self.rank = CrossAttentionFeedForward(hidden_dim,hidden_dim,1)
    def forward(self, batch1):
        # Feature extraction from sequences
        outputs = self.pretrain_model(**batch1)
        # ipdb.set_trace()
        attention_mask = batch1.get("attention_mask")
        features1 = outputs.last_hidden_state # Assuming last layer output
        regression_output1 = self.residualcnn_reg(features1, attention_mask)
        all_layer_features = torch.stack(list(outputs.hidden_states)[1:], dim=-1)#  torch.cat()
        all_layer_pooled = _pool_all_layers(all_layer_features, attention_mask, self.residualcnn_reg)
        return regression_output1, features1, all_layer_features, None, None, all_layer_pooled


class RegModel_v2(nn.Module):
    '''
    remain modification
    '''
    def __init__(self, pretrain_model,config): # cfg.model.rank.residual
        super(RegModel_v2, self).__init__()
        self.pretrain_model = pretrain_model
        # Assume esm_model outputs features of size N
        # self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        # Regression task layers
        
        
        self.residualcnn_reg =ClassificationHead2(config)
        
        # self.regression1 = FeedForward(hidden_dim,hidden_dim,1)
        # self.regression2 = FeedForward(hidden_dim,hidden_dim,1)
        # self.rank = CrossAttentionFeedForward(hidden_dim,hidden_dim,1)
    def forward(self, batch1):
        # Feature extraction from sequences
        outputs = self.pretrain_model(**batch1)
        # ipdb.set_trace()
        attention_mask = batch1["attention_mask"]
        features1 = outputs.last_hidden_state # Assuming last layer output
        regression_output1 = self.residualcnn_reg(features1,attention_mask)
        all_layer_features = torch.stack(list(outputs.hidden_states)[1:], dim=-1)#  torch.cat()
        all_layer_pooled = _pool_all_layers(all_layer_features, attention_mask, self.residualcnn_reg)
        return regression_output1, features1, all_layer_features, None, None, all_layer_pooled


class RegModel_MLTP(nn.Module):
    """
    Multi-Layer Trainable Pooling (MLTP) regression head for LLM backbones.

    It pools token representations by first fusing multiple hidden layers with
    learnable layer weights, then applying a trainable token-level attention
    pooling to produce a single sequence embedding.
    """

    def __init__(self, pretrain_model, config):
        super().__init__()
        self.pretrain_model = pretrain_model

        num_layers = getattr(config, "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError(
                "RegModel_MLTP requires config.num_hidden_layers to be present "
                "(so it knows how many transformer layers to pool)."
            )

        mltp_num_heads = getattr(config, "mltp_num_heads", getattr(config, "num_heads", 8))
        dropout_prob = float(getattr(config, "hidden_dropout_prob", 0.1))

        self.mltp_pooler = MultiLayerTrainablePooling(
            hidden_size=config.hidden_size,
            num_layers=int(num_layers),
            num_heads=int(mltp_num_heads),
            dropout=dropout_prob,
        )

        # For orthogonal constraint/debugging we still pool each layer into a
        # sequence embedding; we must use a token-level pooling that works
        # with apply_pooling(), so we use 'mean' here regardless of mltp.
        config_mean_pool = copy.deepcopy(config)
        config_mean_pool.pooling = "mean"
        self.layer_pool_head = ClassificationHead2(config_mean_pool)

        # Reuse the same regression projection stack as ClassificationHead2(mean).
        self.dropout = self.layer_pool_head.dropout
        self.dense = self.layer_pool_head.dense
        self.out_proj = self.layer_pool_head.out_proj

    def forward(self, batch1):
        outputs = self.pretrain_model(**batch1)
        attention_mask = batch1["attention_mask"]
        features1 = outputs.last_hidden_state

        # [B, T, D, L] where L=num_hidden_layers
        all_layer_features = torch.stack(list(outputs.hidden_states)[1:], dim=-1)

        # [B, D]
        pooled = self.mltp_pooler(all_layer_features, attention_mask)

        x = self.dropout(pooled)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        regression_output1 = self.out_proj(x)

        # [B, D, L] used by Trainer's orthogonal constraint.
        all_layer_pooled = _pool_all_layers(all_layer_features, attention_mask, self.layer_pool_head)

        return regression_output1, features1, all_layer_features, None, None, all_layer_pooled
