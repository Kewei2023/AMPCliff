from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .MultiLayersTrainablePooling import PerceiverResampler

MLTP_PRESETS: Dict[str, Dict[str, Any]] = {
    "esm2_t6": {
        "hidden_size": 320,
        "num_layers": 6,
        "attention_mode": "bidirectional",
        "num_cross_heads": 32,
        "cross_dim_head": 160,
        "num_latents": 60,
        "latent_dim": 320,
        "ffn_mult": 4,
    },
    "esm2_t12": {
        "hidden_size": 480,
        "num_layers": 12,
        "attention_mode": "bidirectional",
        "num_cross_heads": 32,
        "cross_dim_head": 240,
        "num_latents": 90,
        "latent_dim": 480,
        "ffn_mult": 4,
    },
}

_MLTP_OVERRIDE_KEYS = (
    "num_cross_heads",
    "cross_dim_head",
    "num_latents",
    "latent_dim",
    "ffn_mult",
    "normalize_output",
)


def resolve_mltp_method_kwargs(reg_cfg: Any) -> Dict[str, Any]:
    """Read only pooling_config.mltp_paper overrides."""
    pooling_config = getattr(reg_cfg, "pooling_config", None)
    if pooling_config is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(pooling_config):
            pooling_config = OmegaConf.to_container(pooling_config, resolve=True)
    except Exception:
        pass
    if not isinstance(pooling_config, dict):
        return {}
    method = pooling_config.get("mltp_paper", {}) or {}
    if not isinstance(method, dict):
        return {}
    return {
        k: v for k, v in method.items()
        if k in _MLTP_OVERRIDE_KEYS and v is not None
    }


def resolve_mltp_kwargs(
    version: Optional[str],
    config: Any,
    pooling_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if version is None:
        raise ValueError(
            "pooling='mltp_paper' requires model.regression.version "
            "(supported presets: esm2_t6, esm2_t12)."
        )
    if version not in MLTP_PRESETS:
        raise ValueError(
            f"Unknown MLTP preset for version={version!r}. "
            f"Supported: {sorted(MLTP_PRESETS)}."
        )

    preset = dict(MLTP_PRESETS[version])
    hidden_size = getattr(config, "hidden_size", None)
    num_layers = getattr(config, "num_hidden_layers", None)
    if hidden_size != preset["hidden_size"]:
        raise ValueError(
            f"MLTP preset {version!r} expects hidden_size={preset['hidden_size']}, "
            f"got {hidden_size}."
        )
    if num_layers != preset["num_layers"]:
        raise ValueError(
            f"MLTP preset {version!r} expects num_hidden_layers={preset['num_layers']}, "
            f"got {num_layers}."
        )

    resolved = dict(preset)
    resolved["normalize_output"] = True
    pooling_kwargs = pooling_kwargs or {}
    for key in _MLTP_OVERRIDE_KEYS:
        if key in pooling_kwargs and pooling_kwargs[key] is not None:
            resolved[key] = pooling_kwargs[key]
    return resolved


def per_layer_masked_mean(
    all_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    x = all_hidden_states.permute(0, 3, 1, 2)
    if attention_mask is not None:
        mask = attention_mask[:, None, :, None].to(x.dtype)
        return (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=eps)
    return x.mean(dim=2)


class OfficialMLTPPooling(nn.Module):
    """Official MLTP: per-layer masked mean + PerceiverResampler cross-layer attention."""

    def __init__(
        self,
        version: Optional[str],
        config: Any,
        pooling_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        resolved = resolve_mltp_kwargs(version, config, pooling_kwargs)
        self.hidden_size = int(resolved["hidden_size"])
        self.num_layers = int(resolved["num_layers"])
        self.normalize_output = bool(resolved.get("normalize_output", True))

        self.resampler = PerceiverResampler(
            dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            latent_dim=int(resolved["latent_dim"]),
            num_latents_value=int(resolved["num_latents"]),
            num_cross_heads=int(resolved["num_cross_heads"]),
            cross_dim_head=int(resolved["cross_dim_head"]),
            layers=self.num_layers,
            ffn_mult=int(resolved.get("ffn_mult", 4)),
        )
        self.resampler.normalize = self.normalize_output

    def forward(
        self,
        all_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_input_layers = all_hidden_states.shape[-1]
        if num_input_layers != self.num_layers:
            raise ValueError(
                f"OfficialMLTPPooling expected {self.num_layers} hidden layers, "
                f"got {num_input_layers}."
            )
        layer_repr = per_layer_masked_mean(all_hidden_states, attention_mask)
        return self.resampler(layer_repr)


def masked_mean_pooling(
    features: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    eps: float = 1e-9,
) -> torch.Tensor:
    if attention_mask is None:
        return features.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(features.dtype)
    return (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=eps)


def masked_max_pooling(features: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return features.max(dim=1).values
    mask = attention_mask.unsqueeze(-1).bool()
    masked = features.masked_fill(~mask, torch.finfo(features.dtype).min)
    return masked.max(dim=1).values


def last_token_pooling(features: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return features[:, -1, :]
    lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(features.size(0), device=features.device)
    return features[batch_idx, lengths, :]


class LatentAttentionPooling(nn.Module):
    def __init__(self, d_model: int, num_latents: int = 8, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = features.size(0)
        queries = self.latents.unsqueeze(0).expand(B, -1, -1)
        key_padding_mask = None if attention_mask is None else ~attention_mask.bool()
        pooled, _ = self.attn(
            queries, features, features, key_padding_mask=key_padding_mask, need_weights=False
        )
        pooled = pooled.mean(dim=1)
        return self.out_proj(self.dropout(pooled))


class StructuredSelfAttentivePooling(nn.Module):
    """Multi-hop structured self-attention pooling (Lin et al., ICLR 2017)."""

    def __init__(
        self,
        hidden_size: int,
        attention_size: int = 350,
        attention_hops: int = 30,
        dropout: float = 0.5,
        use_bias: bool = False,
        hop_output: str = "flatten",
    ):
        super().__init__()
        hop_output = str(hop_output).lower()
        if hop_output != "flatten":
            raise ValueError(
                f"Unsupported hop_output={hop_output!r}; only 'flatten' is implemented"
            )

        self.hidden_size = int(hidden_size)
        self.attention_size = int(attention_size)
        self.attention_hops = int(attention_hops)
        self.hop_output = hop_output
        self.use_bias = bool(use_bias)

        self.ws1 = nn.Linear(hidden_size, attention_size, bias=use_bias)
        self.ws2 = nn.Linear(attention_size, attention_hops, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(
            attention_hops * hidden_size,
            hidden_size,
            bias=use_bias,
        )
        self._last_attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = features.size(0)
        hbar = torch.tanh(self.ws1(self.dropout(features)))
        logits = self.ws2(hbar)
        alphas = logits.transpose(1, 2)

        if attention_mask is not None:
            alphas = alphas.masked_fill(
                ~attention_mask.unsqueeze(1).bool(),
                torch.finfo(alphas.dtype).min,
            )
        alphas = F.softmax(alphas, dim=-1)
        self._last_attention_weights = alphas

        hop_vecs = torch.bmm(alphas, features)
        pooled = self.out_proj(hop_vecs.reshape(batch_size, -1))
        return pooled

    def compute_penalty_loss(self, coeff: float = 1.0) -> torch.Tensor:
        from ...utils.attention_penalty import structured_attention_penalty

        if self._last_attention_weights is None:
            return torch.tensor(0.0)
        return structured_attention_penalty(self._last_attention_weights) * float(coeff)


def log_revised_pooling_info(pooling_config: str, pooling_module: nn.Module) -> None:
    from ...utils.std_logger import Logger

    param_count = sum(p.numel() for p in pooling_module.parameters() if p.requires_grad)
    impl_status = (
        "revised_reference"
        if pooling_config in {"mltp_paper", "attn_structured"}
        else "legacy"
    )
    Logger.info(f"Pooling config: {pooling_config}")
    Logger.info(f"Pooling class: {pooling_module.__class__.__name__}")
    Logger.info(f"Implementation status: {impl_status}")
    Logger.info(f"Pooling parameter count: {param_count}")


__all__ = [
    "last_token_pooling",
    "LatentAttentionPooling",
    "OfficialMLTPPooling",
    "MLTP_PRESETS",
    "resolve_mltp_kwargs",
    "resolve_mltp_method_kwargs",
    "per_layer_masked_mean",
    "StructuredSelfAttentivePooling",
    "log_revised_pooling_info",
    "masked_mean_pooling",
    "masked_max_pooling",
]
