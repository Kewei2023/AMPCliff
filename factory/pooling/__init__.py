from .attention import MultiHeadVectorAttnPooling
from .local_spectral_anchor_new import MultiHeadLocalSpectralAnchorPoolingNew
from .local_spectrual import MultiHeadLocalSTFTPooling
from .registry import (
    apply_pooling,
    build_pooling_modules,
    get_supported_poolings,
    resolve_pooling_kwargs,
    validate_pooling_name,
)
from .spectral_anchor import SpectralAnchorPooling
from .freq_scale_seq import LearnableFreqScaleSeq, LearnableDCACScaleSeq
__all__ = [
    "SpectralAnchorPooling",
    "MultiHeadVectorAttnPooling",
    "MultiHeadLocalSpectralAnchorPoolingNew",
    "MultiHeadLocalSTFTPooling",
    "apply_pooling",
    "build_pooling_modules",
    "get_supported_poolings",
    "resolve_pooling_kwargs",
    "validate_pooling_name",
    "LearnableFreqScaleSeq",
    "LearnableDCACScaleSeq",
]
