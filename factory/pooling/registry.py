# maintained by kewei li
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ...utils.std_logger import Logger
from .spectral_anchor import SpectralAnchorPooling
from .spectral_anchor_v2 import (
    FFTLatentAttentionGatePooling,
    FrequencyAwarePooling,
    MultiHeadLocalSpectralAnchorPooling,
    MultiHeadSpectralAnchorPooling,
    SEPooling,
    SpectralAnchorPoolingV2,
    STFTLatentAttentionMaxPooling,
)
from .local_spectrual import MultiHeadLocalSTFTPooling
from .local_spectrual_v2 import MultiScaleAnchorFreeSTFTPoolingV2
from .local_spectral_anchor_new import MultiHeadLocalSpectralAnchorPoolingNew
from .llm_pooling_dropin import (
    LatentAttentionPooling,
    SlicedWassersteinPooling,
    last_token_pooling,
)

SUPPORTED_POOLINGS: Tuple[str, ...] = (
    "mean",
    "max",
    "attn",
    "last",
    "latent_attn",
    "swe_ot",
    "mltp",
    "spectral_anchor",
    "spectral_anchor_v2",
    "multi_head_spectral",
    "local_spectral_anchor",
    "mlsap",
    "local_spectral_anchor_new",
    "local_stft",
    "local_stft_v2",
    "stft_latent_attn_max",
    "fft_latent_attn_gate",
    "frequency_aware",
    "se_pooling",
    # Ablation variants of FFT-LAG (all route to FFTLatentAttentionGatePooling)
)

# Aggregation strategies for spectral_anchor_v2
AGGREGATION_STRATEGIES: Tuple[str, ...] = ("soft", "max", "mean")


def _coerce_int_list(val: Any, *, context: str = "int_list") -> List[int]:
    """Normalize YAML/OmegaConf list/tuple to list[int]."""
    if val is None:
        raise ValueError(f"{context} must not be None")
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    try:
        from omegaconf import ListConfig

        if isinstance(val, ListConfig):
            return [int(x) for x in list(val)]
    except ImportError:
        pass
    raise TypeError(f"{context}: expected list of ints, got {type(val)!r}")


def _coerce_optional_int_list(val: Any) -> Optional[List[int]]:
    if val is None:
        return None
    return _coerce_int_list(val, context="optional_int_list")


def _coerce_bool(val: Any, default: bool = True) -> bool:
    """CLI/YAML may pass strings; bool('false') is True in Python — normalize."""
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




# Keyword names accepted by build_pooling_modules (excluding pooling, d_model, attn_factory).
_BUILD_POOLING_MODULE_KWARGS: frozenset = frozenset(
    {
        "num_anchor",
        "use_fft",
        "aggregation",
        "use_projection",
        "num_heads",
        "num_freq_components",
        "num_latents",
        "num_projections",
        "num_quantiles",
        "gated",
        "dropout",
        "analysis_dim",
        "stft_n_fft",
        "stft_win_length",
        "stft_hop_length",
        "stft_center",
        "use_phase",
        "scorer_type",
        "scorer_hidden_dim",
        "frame_weight_version",
        "local_stft_v2_n_ffts",
        "local_stft_v2_win_lengths",
        "local_stft_v2_hop_lengths",
        "local_stft_v2_frame_temperature",
        "local_stft_v2_output_proj",
        "local_stft_v2_residual_mix_init",
        "stft_latent_attn_num_heads",
        "stft_latent_use_complex_residual",
        "stft_latent_return_time_gate",
        "stft_latent_token_gate_mode",
        "stft_latent_eps",
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
    "local_stft_v2": {
        "n_ffts": "local_stft_v2_n_ffts",
        "win_lengths": "local_stft_v2_win_lengths",
        "hop_lengths": "local_stft_v2_hop_lengths",
        "frame_temperature": "local_stft_v2_frame_temperature",
        "output_proj": "local_stft_v2_output_proj",
        "residual_mix_init": "local_stft_v2_residual_mix_init",
    },
    "stft_latent_attn_max": {
        "attn_num_heads": "stft_latent_attn_num_heads",
        "use_complex_residual": "stft_latent_use_complex_residual",
        "return_time_gate": "stft_latent_return_time_gate",
        "token_gate_mode": "stft_latent_token_gate_mode",
        "eps": "stft_latent_eps",
    },
    "fft_latent_attn_gate": dict(_FFT_LATENT_SHORT_RENAME),
    # Ablation variants share the same short-key rename mapping
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
    """Support pre-refactor flat Hydra keys on model.regression / model.classification."""
    merged: Dict[str, Any] = {}
    for k in _BUILD_POOLING_MODULE_KWARGS:
        if _reg_cfg_has_key(reg_cfg, k):
            merged[k] = _reg_cfg_get(reg_cfg, k)
    return merged


def resolve_pooling_kwargs(reg_cfg: Any) -> Dict[str, Any]:
    """
    Merge pooling_common + pooling_config[pooling] (with per-method short-key renames)
    into a flat dict for build_pooling_modules(...).

    If pooling_common / pooling_config are absent, falls back to legacy flat keys on reg_cfg.
    """
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

    # CLI / legacy flat overrides on model.regression.* (sibling keys) win over nested defaults.
    for k in _BUILD_POOLING_MODULE_KWARGS:
        if _reg_cfg_has_key(reg_cfg, k):
            merged[k] = _reg_cfg_get(reg_cfg, k)

    return {k: v for k, v in merged.items() if k in _BUILD_POOLING_MODULE_KWARGS}


def build_pooling_modules(
    pooling: str,
    d_model: int,
    *,
    num_anchor: int = 8,
    use_fft: bool = True,
    # New parameters for v2
    aggregation: str = "soft",
    use_projection: bool = True,
    num_heads: int = 4,
    num_freq_components: int = 16,
    # LLM readout-style pooling params
    num_latents: int = 8,
    num_projections: int = 32,
    num_quantiles: int = 8,
    gated: bool = True,
    dropout: float = 0.0,
    # local_spectral_anchor (MultiHeadLocalSpectralAnchorPooling)
    analysis_dim: int = 8,
    stft_n_fft: int = 8,
    stft_win_length: Optional[int] = None,
    stft_hop_length: Optional[int] = None,
    stft_center: bool = False,
    use_phase: bool = False,
    scorer_type: str = "mlp",
    scorer_hidden_dim: Optional[int] = None,
    frame_weight_version: str = "old",
    # local_stft_v2 (MultiScaleAnchorFreeSTFTPoolingV2)
    local_stft_v2_n_ffts: Any = None,
    local_stft_v2_win_lengths: Any = None,
    local_stft_v2_hop_lengths: Any = None,
    local_stft_v2_frame_temperature: float = 0.5,
    local_stft_v2_output_proj: bool = True,
    local_stft_v2_residual_mix_init: float = 0.1,
    # stft_latent_attn_max (STFTLatentAttentionMaxPooling): spectral latent attention + gated max
    stft_latent_attn_num_heads: int = 4,
    stft_latent_use_complex_residual: bool = True,
    stft_latent_return_time_gate: bool = False,
    stft_latent_token_gate_mode: str = "scalar",
    stft_latent_eps: float = 1e-6,
    # fft_latent_attn_gate / fft_latent_attn_max (FFTLatentAttentionGatePooling):
    # rFFT latent attention + gate modulation
    fft_latent_gate_num_heads: int = 4,
    fft_latent_gate_num_latents: Optional[int] = None,
    fft_latent_gate_time_pool: Optional[str] = None,
    fft_latent_gate_residual: bool = True,
    fft_latent_gate_eps: float = 1e-6,
    fft_latent_gate_dropout: Optional[float] = None,
    fft_latent_gate_use_gate: bool = True,
    fft_latent_gate_use_latent: bool = True,
    # Existing parameters
    attn_factory: Optional[Callable[[int], nn.Module]] = None,
) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    """
    Build pooling modules based on the specified pooling type.

    Args:
        pooling: Pooling type name
        d_model: Hidden dimension
        num_anchor: Number of anchors for spectral methods
        use_fft: Whether to use FFT for spectral methods (ignored for local_spectral_anchor, which always uses STFT)
        aggregation: Aggregation strategy for spectral_anchor_v2 ('soft', 'max', 'mean')
        use_projection: Whether to use input/output projections for spectral_anchor_v2
        num_heads: Number of heads for multi_head_spectral
        num_freq_components: Number of frequency components for frequency_aware
        gated: Whether to use gating mechanism
        dropout: Dropout rate
        analysis_dim: Per-head STFT analysis channels for local_spectral_anchor
        stft_n_fft: FFT size for local_spectral_anchor STFT
        stft_win_length: Window length (defaults to stft_n_fft if None)
        stft_hop_length: Hop length (defaults to max(1, win_length//2) if None)
        stft_center: If True, use STFT center padding (see torch.stft)
        use_phase: If True, concatenate real/imag STFT bins in frame descriptors
        frame_weight_version: Deprecated for local_spectral_anchor. Use
            pooling='local_spectral_anchor_new' to enable new behavior.
        attn_factory: Factory function for attention pooling

    Returns:
        Tuple of (attn_pool, sap_pool) modules
    """
    pooling = validate_pooling_name(pooling)
    attn_pool = None
    sap_pool = None

    use_fft = _coerce_bool(use_fft, True)
    use_projection = _coerce_bool(use_projection, True)
    gated = _coerce_bool(gated, True)
    stft_center = _coerce_bool(stft_center, False)
    use_phase = _coerce_bool(use_phase, False)

    if pooling == "attn":
        if attn_factory is None:
            raise ValueError("attn_factory is required when pooling='attn'")
        attn_pool = attn_factory(d_model)
    elif pooling == "spectral_anchor":
        sap_pool = SpectralAnchorPooling(
            d_model=d_model,
            num_anchor=num_anchor,
            use_fft=use_fft,
        )
    elif pooling == "spectral_anchor_v2":
        if aggregation not in AGGREGATION_STRATEGIES:
            raise ValueError(
                f"aggregation must be one of {list(AGGREGATION_STRATEGIES)}, got '{aggregation}'"
            )
        sap_pool = SpectralAnchorPoolingV2(
            d_model=d_model,
            num_anchor=num_anchor,
            use_fft=use_fft,
            aggregation=aggregation,
            use_projection=use_projection,
        )
    elif pooling == "multi_head_spectral":
        sap_pool = MultiHeadSpectralAnchorPooling(
            d_model=d_model,
            num_heads=num_heads,
            num_anchor_per_head=num_anchor // num_heads if num_anchor >= num_heads else 2,
            use_fft=use_fft,
            gated=gated,
            dropout=dropout,
        )
    elif pooling in ("local_spectral_anchor", "mlsap"):
        if not use_fft:
            Logger.warning(
                f"pooling='{pooling}' always uses STFT; "
                "use_fft=False is ignored (use_fft applies to multi_head_spectral / spectral_anchor*)."
            )
        local_common_kwargs = dict(
            d_model=d_model,
            num_heads=num_heads,
            num_anchor_per_head=num_anchor // num_heads if num_anchor >= num_heads else 2,
            analysis_dim=int(analysis_dim),
            n_fft=int(stft_n_fft),
            win_length=stft_win_length,
            hop_length=stft_hop_length,
            stft_center=stft_center,
            use_phase=use_phase,
            gated=gated,
            dropout=dropout,
        )
        frame_weight_version = (
            "old" if frame_weight_version is None else str(frame_weight_version).lower()
        )
        if frame_weight_version not in ("old", "new"):
            raise ValueError(
                "frame_weight_version must be one of {'old','new'} for "
                "pooling='local_spectral_anchor', "
                f"got '{frame_weight_version}'"
            )
        if frame_weight_version == "new":
            raise ValueError(
                "frame_weight_version='new' is not supported for pooling='local_spectral_anchor'. "
                "Use pooling='local_spectral_anchor_new' for explicit new behavior."
            )
        sap_pool = MultiHeadLocalSpectralAnchorPooling(**local_common_kwargs)
    elif pooling == "local_spectral_anchor_new":
        if not use_fft:
            Logger.warning(
                "pooling='local_spectral_anchor_new' always uses STFT; "
                "use_fft=False is ignored (use_fft applies to multi_head_spectral / spectral_anchor*)."
            )
        sap_pool = MultiHeadLocalSpectralAnchorPoolingNew(
            d_model=d_model,
            num_heads=num_heads,
            num_anchor_per_head=num_anchor // num_heads if num_anchor >= num_heads else 2,
            analysis_dim=int(analysis_dim),
            n_fft=int(stft_n_fft),
            win_length=stft_win_length,
            hop_length=stft_hop_length,
            stft_center=stft_center,
            use_phase=use_phase,
            gated=gated,
            dropout=dropout,
        )
    elif pooling == "local_stft":
        if not use_fft:
            Logger.warning(
                "pooling='local_stft' always uses STFT; "
                "use_fft=False is ignored (use_fft applies to multi_head_spectral / spectral_anchor*)."
            )
        sap_pool = MultiHeadLocalSTFTPooling(
            d_model=d_model,
            num_heads=num_heads,
            analysis_dim=int(analysis_dim),
            n_fft=int(stft_n_fft),
            win_length=stft_win_length,
            hop_length=stft_hop_length,
            stft_center=stft_center,
            use_phase=use_phase,
            scorer_type=scorer_type,
            scorer_hidden_dim=scorer_hidden_dim,
            gated=gated,
            dropout=dropout,
        )
    elif pooling == "local_stft_v2":
        if not use_fft:
            Logger.warning(
                "pooling='local_stft_v2' always uses STFT; "
                "use_fft=False is ignored (use_fft applies to multi_head_spectral / spectral_anchor*)."
            )
        n_ffts_list = (
            _coerce_int_list(local_stft_v2_n_ffts, context="local_stft_v2_n_ffts")
            if local_stft_v2_n_ffts is not None
            else [4, 8]
        )
        win_list = _coerce_optional_int_list(local_stft_v2_win_lengths)
        hop_list = _coerce_optional_int_list(local_stft_v2_hop_lengths)
        local_stft_v2_output_proj_b = _coerce_bool(local_stft_v2_output_proj, True)
        sap_pool = MultiScaleAnchorFreeSTFTPoolingV2(
            d_model=d_model,
            num_heads=num_heads,
            analysis_dim=int(analysis_dim),
            n_ffts=n_ffts_list,
            win_lengths=win_list,
            hop_lengths=hop_list,
            stft_center=stft_center,
            use_phase=use_phase,
            scorer_hidden_dim=scorer_hidden_dim,
            frame_temperature=float(local_stft_v2_frame_temperature),
            gated=gated,
            output_proj=local_stft_v2_output_proj_b,
            residual_mix_init=float(local_stft_v2_residual_mix_init),
            dropout=float(dropout),
        )
    elif pooling == "stft_latent_attn_max":
        if not use_fft:
            Logger.warning(
                "pooling='stft_latent_attn_max' always uses STFT; "
                "use_fft=False is ignored (use_fft applies to multi_head_spectral / spectral_anchor*)."
            )
        stft_latent_use_complex_residual_b = _coerce_bool(
            stft_latent_use_complex_residual, True
        )
        stft_latent_return_time_gate_b = _coerce_bool(
            stft_latent_return_time_gate, False
        )
        token_mode = (
            str(stft_latent_token_gate_mode).strip().lower()
            if stft_latent_token_gate_mode is not None
            else "scalar"
        )
        if token_mode not in ("scalar", "vector"):
            raise ValueError(
                "stft_latent_token_gate_mode must be 'scalar' or 'vector', "
                f"got '{stft_latent_token_gate_mode}'"
            )
        sap_pool = STFTLatentAttentionMaxPooling(
            d_model=d_model,
            analysis_dim=int(analysis_dim),
            num_latents=int(num_latents),
            num_heads=int(stft_latent_attn_num_heads),
            n_fft=int(stft_n_fft),
            win_length=stft_win_length,
            hop_length=stft_hop_length,
            dropout=float(dropout),
            stft_center=stft_center,
            use_complex_residual=stft_latent_use_complex_residual_b,
            return_time_gate=stft_latent_return_time_gate_b,
            token_gate_mode=token_mode,
            eps=float(stft_latent_eps),
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
    elif pooling == "frequency_aware":
        sap_pool = FrequencyAwarePooling(
            d_model=d_model,
            num_freq_components=num_freq_components,
            combine_temporal=True,
        )
    elif pooling == "se_pooling":
        sap_pool = SEPooling(d_model=d_model, dropout=float(dropout))
    elif pooling == "latent_attn":
        # Trainable latent attention pooling over token embeddings.
        # Implemented as attn_pool to reuse apply_pooling() routing.
        attn_pool = LatentAttentionPooling(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=num_heads,
            dropout=dropout,
        )
    elif pooling == "swe_ot":
        # Lightweight distributional pooling approximation.
        sap_pool = SlicedWassersteinPooling(
            d_model=d_model,
            num_projections=num_projections,
            num_quantiles=num_quantiles,
            dropout=dropout,
        )
    elif pooling == "mltp":
        # Special case: multi-layer pooling needs hidden states, so it cannot
        # be executed by token-level apply_pooling() used in ClassificationHead*.
        # RegModel_MLTP is responsible for wiring.
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
    """
    Apply pooling to features.

    Args:
        features: Input tensor of shape (batch, seq_len, d_model)
        attention_mask: Optional attention mask
        pooling: Pooling type name
        mean_pooling: Function for mean pooling
        max_pooling: Function for max pooling
        attn_pool: Attention pooling module
        sap_pool: Spectral anchor pooling module

    Returns:
        Pooled tensor of shape (batch, d_model)
    """
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
    if pooling == "latent_attn":
        if attn_pool is None:
            raise RuntimeError("attn_pool is not initialized for pooling='latent_attn'")
        return attn_pool(features, attention_mask)
    if pooling == "swe_ot":
        if sap_pool is None:
            raise RuntimeError("sap_pool is not initialized for pooling='swe_ot'")
        return sap_pool(features, attention_mask)
    if pooling == "mltp":
        raise ValueError(
            "pooling='mltp' requires RegModel_MLTP (multi-layer pooling) and cannot be used "
            "as a token-level pooling inside ClassificationHead*."
        )
    if pooling in (
        "spectral_anchor",
        "spectral_anchor_v2",
        "multi_head_spectral",
        "local_spectral_anchor",
        "mlsap",
        "local_spectral_anchor_new",
        "local_stft",
        "local_stft_v2",
        "stft_latent_attn_max",
        "fft_latent_attn_gate",
                                    "frequency_aware",
        "se_pooling",
                                                        ):
        if sap_pool is None:
            raise RuntimeError(f"sap_pool is not initialized for pooling='{pooling}'")
        return sap_pool(features, attention_mask)

    raise ValueError(f"Unsupported pooling: {pooling}")
