import torch
import torch.nn as nn
from typing import Optional
from AMPCliff.spectrual_filter.filter import dct_ortho, idct_ortho


    
class LearnableDCACScaleSeq(nn.Module):
    """
    Two-parameter version:
    - one learnable scale for DC
    - one learnable scale for all AC
    Supports both FFT and DCT.
    """

    def __init__(
        self,
        mode: str = "fft",
        dc_init: float = 1.0,
        ac_init: float = 1.0,
        scale_range: float = 2.0,
    ):
        super().__init__()
        if mode.lower() not in {"fft", "dct"}:
            raise ValueError(f"mode must be 'fft' or 'dct', got {mode}")

        self.mode = mode.lower()
        self.scale_range = float(scale_range)

        dc_prob = min(max(dc_init / scale_range, 1e-4), 1.0 - 1e-4)
        ac_prob = min(max(ac_init / scale_range, 1e-4), 1.0 - 1e-4)

        self.dc_logit = nn.Parameter(torch.tensor(torch.logit(torch.tensor(dc_prob)).item()))
        self.ac_logit = nn.Parameter(torch.tensor(torch.logit(torch.tensor(ac_prob)).item()))

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = features.shape

        x = features
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)

        dc_scale = self.scale_range * torch.sigmoid(self.dc_logit)
        ac_scale = self.scale_range * torch.sigmoid(self.ac_logit)

        if self.mode == "fft":
            coeff = torch.fft.rfft(x, dim=1, n=T, norm="backward")
            coeff = coeff.clone()
            coeff[:, 0, :] = coeff[:, 0, :] * dc_scale
            if coeff.size(1) > 1:
                coeff[:, 1:, :] = coeff[:, 1:, :] * ac_scale
            x_out = torch.fft.irfft(coeff, n=T, dim=1, norm="backward")

        else:  # dct
            coeff = dct_ortho(x, dim=1)
            coeff = coeff.clone()
            coeff[:, 0, :] = coeff[:, 0, :] * dc_scale
            if coeff.size(1) > 1:
                coeff[:, 1:, :] = coeff[:, 1:, :] * ac_scale
            x_out = idct_ortho(coeff, dim=1)

        if attention_mask is not None:
            x_out = x_out * attention_mask.unsqueeze(-1).to(x_out.dtype)

        return x_out
    
class LearnableFreqScaleSeq(nn.Module):
    """
    Minimal learnable frequency-amplitude scaling over sequence axis.

    mode = 'fft':
        x -> rfft(seq_len) -> learnable per-frequency scaling -> irfft(seq_len)

    mode = 'dct':
        x -> dct(seq_len)  -> learnable per-frequency scaling -> idct(seq_len)

    Notes
    -----
    1. No attention.
    2. No latent.
    3. No concat(real, imag).
    4. No linear mixing over hidden dimension.
    5. Scaling is frequency-wise and shared across hidden channels.
    """

    def __init__(
        self,
        max_seq_len: int,
        mode: str = "fft",           # "fft" or "dct"
        init_scale: float = 1.0,
        scale_range: float = 2.0,    # final scale in (0, scale_range)
        use_residual: bool = False,  # if True: scale = 1 + gated_scale
    ):
        super().__init__()

        if mode.lower() not in {"fft", "dct"}:
            raise ValueError(f"mode must be 'fft' or 'dct', got {mode}")

        self.max_seq_len = int(max_seq_len)
        self.mode = mode.lower()
        self.scale_range = float(scale_range)
        self.use_residual = bool(use_residual)

        if self.mode == "fft":
            self.max_freq_bins = self.max_seq_len // 2 + 1
        else:
            self.max_freq_bins = self.max_seq_len

        # use logits, so scale stays bounded and stable
        init_prob = float(init_scale) / float(scale_range)
        init_prob = min(max(init_prob, 1e-4), 1.0 - 1e-4)
        init_logit = torch.logit(torch.tensor(init_prob, dtype=torch.float32))

        self.freq_logits = nn.Parameter(
            torch.full((self.max_freq_bins,), init_logit.item(), dtype=torch.float32)
        )

    def _get_scale(self, n_freq: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        scale = self.scale_range * torch.sigmoid(self.freq_logits[:n_freq])
        if self.use_residual:
            scale = 1.0 + scale
        return scale.to(device=device, dtype=dtype).view(1, n_freq, 1)  # (1, F, 1)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        features: (B, T, D)
        returns:  (B, T, D)
        """
        B, T, D = features.shape

        x = features
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)

        if self.mode == "fft":
            coeff = torch.fft.rfft(x, dim=1, n=T, norm="backward")     # (B, F, D), complex
            scale = self._get_scale(coeff.size(1), coeff.real.dtype, coeff.device)
            coeff = coeff * scale
            x_out = torch.fft.irfft(coeff, n=T, dim=1, norm="backward")  # (B, T, D)

        else:  # dct
            coeff = dct_ortho(x, dim=1)                                 # (B, T, D), real
            scale = self._get_scale(coeff.size(1), coeff.dtype, coeff.device)
            coeff = coeff * scale
            x_out = idct_ortho(coeff, dim=1)                            # (B, T, D)

        if attention_mask is not None:
            x_out = x_out * attention_mask.unsqueeze(-1).to(x_out.dtype)

        return x_out