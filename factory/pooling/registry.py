from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .flag_pooling import FFTLatentAttentionGatePooling
from .llm_pooling_dropin import (
    LatentAttentionPooling,
    StructuredSelfAttentivePooling,
    last_token_pooling,
)

SUPPORTED_POOLINGS: Tuple[str, ...] = (
    "mean",
    "max",
    "attn",
    "last",
    "latent_attn",
    "attn_structured",
    "mltp_paper",
    "fft_latent_attn_gate",
)


def _coerce_bool(val: Any, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("false", "0", "no", "off"):
            return False
        if s in ("true", "1", "yes", "on"):
            return True
        return default
    return bool(val)


def get_supported_poolings() -> Tuple[str, ...]:
    return SUPPORTED_POOLINGS


def validate_pooling_name(
    pooling: str,
    allowed: Optional[Sequence[str]] = None,
    context: str = "pooling",
) -> str:
    allowed_values = tuple(allowed) if allowed is not None else SUPPORTED_POOLINGS
    if pooling not in allowed_values:
        raise ValueError(
            f"{context} must be one of {list(allowed_values)}, got '{pooling}'"
        )
    return pooling


_BUILD_POOLING_MODULE_KWARGS: frozenset = frozenset(
    {
        "num_latents",
        "num_heads",
        "gated",
        "dropout",
        "attention_size",
        "attention_hops",
        "attention_dropout",
        "penalization_coeff",
        "use_bias",
        "hop_output",
        "normalize_output",
        "num_cross_heads",
        "cross_dim_head",
        "latent_dim",
        "ffn_mult",
        "fft_latent_gate_num_heads",
        "fft_latent_gate_num_latents",
        "fft_latent_gate_time_pool",
        "fft_latent_gate_residual",
        "fft_latent_gate_eps",
        "fft_latent_gate_dropout",
        "fft_latent_gate_use_gate",
        "fft_latent_gate_use_latent",
    }
)

_FFT_LATENT_SHORT_RENAME: Dict[str, str] = {
    "num_heads": "fft_latent_gate_num_heads",
    "num_latents": "fft_latent_gate_num_latents",
    "time_pool": "fft_latent_gate_time_pool",
    "gate_residual": "fft_latent_gate_residual",
    "eps": "fft_latent_gate_eps",
    "dropout": "fft_latent_gate_dropout",
    "use_gate": "fft_latent_gate_use_gate",
    "use_latent": "fft_latent_gate_use_latent",
}

_PER_METHOD_SHORT_RENAME: Dict[str, Dict[str, str]] = {
    "fft_latent_attn_gate": dict(_FFT_LATENT_SHORT_RENAME),
}


def _to_plain_mapping(node: Any) -> Dict[str, Any]:
    if node is None:
        return {}
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(node, DictConfig) or OmegaConf.is_config(node):
            out = OmegaConf.to_container(node, resolve=True)
            return dict(out) if isinstance(out, dict) else {}
    except ImportError:
        pass
    if isinstance(node, Mapping):
        return dict(node)
    return {}


def _reg_cfg_has_key(reg_cfg: Any, key: str) -> bool:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(reg_cfg):
            return key in reg_cfg
    except ImportError:
        pass
    return hasattr(reg_cfg, key)


def _reg_cfg_get(reg_cfg: Any, key: str, default: Any = None) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(reg_cfg):
            return reg_cfg.get(key, default)
    except ImportError:
        pass
    return getattr(reg_cfg, key, default)


def _apply_method_short_renames(pooling: str, method_d: Mapping[str, Any]) -> Dict[str, Any]:
    rename = _PER_METHOD_SHORT_RENAME.get(pooling)
    if not rename or not method_d:
        return dict(method_d)
    out: Dict[str, Any] = {}
    for k, v in method_d.items():
        nk = rename.get(str(k), str(k))
        out[nk] = v
    return out


def _legacy_flat_pooling_kwargs(reg_cfg: Any) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for k in _BUILD_POOLING_MODULE_KWARGS:
        if _reg_cfg_has_key(reg_cfg, k):
            merged[k] = _reg_cfg_get(reg_cfg, k)
    return merged


def resolve_pooling_kwargs(reg_cfg: Any) -> Dict[str, Any]:
    pooling_raw = _reg_cfg_get(reg_cfg, "pooling", "mean")
    pooling = str(pooling_raw).strip() if pooling_raw is not None else "mean"

    has_nested = _reg_cfg_has_key(reg_cfg, "pooling_config") or _reg_cfg_has_key(
        reg_cfg, "pooling_common"
    )
    if has_nested:
        common = _to_plain_mapping(_reg_cfg_get(reg_cfg, "pooling_common", None))
        pc = _to_plain_mapping(_reg_cfg_get(reg_cfg, "pooling_config", None))
        method_raw = _to_plain_mapping(pc.get(pooling)) if isinstance(pc, dict) else {}
        method = _apply_method_short_renames(pooling, method_raw)
        merged = {**common, **method}
    else:
        merged = _legacy_flat_pooling_kwargs(reg_cfg)

    for k in _BUILD_POOLING_MODULE_KWARGS:
        if _reg_cfg_has_key(reg_cfg, k):
            merged[k] = _reg_cfg_get(reg_cfg, k)

    return {k: v for k, v in merged.items() if k in _BUILD_POOLING_MODULE_KWARGS}


def build_pooling_modules(
    pooling: str,
    d_model: int,
    *,
    num_latents: int = 8,
    num_heads: int = 4,
    gated: bool = True,
    dropout: float = 0.0,
    attention_size: int = 128,
    attention_hops: int = 30,
    attention_dropout: float = 0.5,
    penalization_coeff: float = 1.0,
    use_bias: bool = False,
    hop_output: str = "flatten",
    fft_latent_gate_num_heads: int = 4,
    fft_latent_gate_num_latents: Optional[int] = None,
    fft_latent_gate_time_pool: Optional[str] = None,
    fft_latent_gate_residual: bool = True,
    fft_latent_gate_eps: float = 1e-6,
    fft_latent_gate_dropout: Optional[float] = None,
    fft_latent_gate_use_gate: bool = True,
    fft_latent_gate_use_latent: bool = True,
    attn_factory: Optional[Callable[[int], nn.Module]] = None,
) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    pooling = validate_pooling_name(pooling)
    attn_pool = None
    sap_pool = None

    gated = _coerce_bool(gated, True)

    if pooling == "attn":
        if attn_factory is None:
            raise ValueError("attn_factory is required when pooling='attn'")
        attn_pool = attn_factory(d_model)
    elif pooling == "attn_structured":
        attn_pool = StructuredSelfAttentivePooling(
            hidden_size=d_model,
            attention_size=int(attention_size),
            attention_hops=int(attention_hops),
            dropout=float(attention_dropout),
            use_bias=_coerce_bool(use_bias, False),
            hop_output=str(hop_output),
        )
    elif pooling == "latent_attn":
        attn_pool = LatentAttentionPooling(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=num_heads,
            dropout=dropout,
        )
    elif pooling == "fft_latent_attn_gate":
        gate_time_pool = (
            str(fft_latent_gate_time_pool).strip().lower()
            if fft_latent_gate_time_pool is not None
            else "max"
        )
        if gate_time_pool not in ("mean", "max", "attn"):
            raise ValueError(
                "fft_latent_gate_time_pool must be 'mean', 'max', or 'attn', "
                f"got '{fft_latent_gate_time_pool}'"
            )
        gate_dropout = (
            float(dropout)
            if fft_latent_gate_dropout is None
            else float(fft_latent_gate_dropout)
        )
        gate_num_latents = (
            int(num_latents)
            if fft_latent_gate_num_latents is None
            else int(fft_latent_gate_num_latents)
        )
        sap_pool = FFTLatentAttentionGatePooling(
            d_model=d_model,
            num_latents=gate_num_latents,
            num_heads=int(fft_latent_gate_num_heads),
            dropout=gate_dropout,
            time_pool=gate_time_pool,
            gate_residual=_coerce_bool(fft_latent_gate_residual, True),
            eps=float(fft_latent_gate_eps),
            use_gate=_coerce_bool(fft_latent_gate_use_gate, True),
            use_latent=_coerce_bool(fft_latent_gate_use_latent, True),
        )
    elif pooling == "mltp_paper":
        pass

    return attn_pool, sap_pool


def apply_pooling(
    features: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pooling: str,
    *,
    mean_pooling: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    max_pooling: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    attn_pool: Optional[nn.Module] = None,
    sap_pool: Optional[nn.Module] = None,
) -> torch.Tensor:
    if features.ndim != 3:
        return features

    pooling = validate_pooling_name(pooling)
    if pooling == "mean":
        return mean_pooling(features, attention_mask)
    if pooling == "max":
        return max_pooling(features, attention_mask)
    if pooling == "last":
        return last_token_pooling(features, attention_mask)
    if pooling == "attn":
        if attn_pool is None:
            raise RuntimeError("attn_pool is not initialized for pooling='attn'")
        return attn_pool(features, attention_mask)
    if pooling == "attn_structured":
        if attn_pool is None:
            raise RuntimeError("attn_pool is not initialized for pooling='attn_structured'")
        return attn_pool(features, attention_mask)
    if pooling == "latent_attn":
        if attn_pool is None:
            raise RuntimeError("attn_pool is not initialized for pooling='latent_attn'")
        return attn_pool(features, attention_mask)
    if pooling == "mltp_paper":
        raise ValueError(
            "pooling='mltp_paper' requires RegModel_MLTP_Paper (multi-layer pooling) and cannot be used "
            "as a token-level pooling inside ClassificationHead*."
        )
    if pooling == "fft_latent_attn_gate":
        if sap_pool is None:
            raise RuntimeError(f"sap_pool is not initialized for pooling='{pooling}'")
        return sap_pool(features, attention_mask)

    raise ValueError(f"Unsupported pooling: {pooling}")
