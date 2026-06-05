import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class MultiHeadLocalSTFTPooling(nn.Module):
    """
    Anchor-free multi-head local STFT pooling for 1D token sequences.

    Pipeline per head:
        x_h
        -> analysis_proj
        -> STFT
        -> frame descriptors
        -> frame scorer (MLP)
        -> frame softmax weights
        -> overlap-aware token weights
        -> weighted sum of original x_h

    Compared with anchor-based spectral pooling:
        - removes prototype / anchor assignment
        - avoids anchor-collapse issue
        - still uses local time-frequency descriptors to produce token weights
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        analysis_dim: int = 8,
        n_fft: int = 8,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        stft_center: bool = False,
        use_phase: bool = False,
        scorer_hidden_dim: Optional[int] = None,
        scorer_type: str = "mlp",   # "linear" or "mlp"
        gated: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.analysis_dim = analysis_dim
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else max(1, self.win_length // 2)
        self.stft_center = bool(stft_center)
        self.use_phase = bool(use_phase)
        self.gated = bool(gated)
        self.eps = eps

        if analysis_dim < 1:
            raise ValueError(f"analysis_dim must be >= 1, got {analysis_dim}")
        if self.n_fft < 1:
            raise ValueError(f"n_fft must be >= 1, got {self.n_fft}")
        if not (1 <= self.win_length <= self.n_fft):
            raise ValueError(
                f"win_length must satisfy 1 <= win_length <= n_fft, got "
                f"win_length={self.win_length}, n_fft={self.n_fft}"
            )
        if self.hop_length < 1:
            raise ValueError(f"hop_length must be >= 1, got {self.hop_length}")

        self.num_freq_bins = self.n_fft // 2 + 1
        if self.use_phase:
            self.frame_desc_dim = self.analysis_dim * self.num_freq_bins * 2
        else:
            self.frame_desc_dim = self.analysis_dim * self.num_freq_bins

        self.analysis_proj = nn.ModuleList(
            [nn.Linear(self.head_dim, self.analysis_dim, bias=True) for _ in range(self.num_heads)]
        )

        self.frame_proj = nn.ModuleList(
            [nn.Linear(self.frame_desc_dim, self.frame_desc_dim, bias=True) for _ in range(self.num_heads)]
        )

        if scorer_hidden_dim is None:
            scorer_hidden_dim = max(16, self.frame_desc_dim // 2)

        self.frame_scorers = nn.ModuleList()
        for _ in range(self.num_heads):
            if scorer_type == "linear":
                scorer = nn.Linear(self.frame_desc_dim, 1, bias=True)
            elif scorer_type == "mlp":
                scorer = nn.Sequential(
                    nn.Linear(self.frame_desc_dim, scorer_hidden_dim),
                    nn.GELU(),
                    nn.Linear(scorer_hidden_dim, 1),
                )
            else:
                raise ValueError(f"Unsupported scorer_type: {scorer_type}")
            self.frame_scorers.append(scorer)

        if self.gated:
            self.gate = nn.Linear(self.d_model, self.d_model)

        self.output_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)

    def _build_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.hann_window(self.win_length, device=device, dtype=dtype)

    def _stft_frame_descriptors(
        self, x_h: torch.Tensor, proj: nn.Linear
    ) -> torch.Tensor:
        """
        x_h: (B, T, head_dim)
        return: (B, n_frames, frame_desc_dim)
        """
        B, T, _ = x_h.shape

        a = proj(x_h)                 # (B, T, analysis_dim)
        a = a.transpose(1, 2)         # (B, analysis_dim, T)
        a_flat = a.reshape(B * self.analysis_dim, T)

        min_len = self.n_fft + 1
        if a_flat.size(-1) < min_len:
            a_flat = F.pad(a_flat, (0, min_len - a_flat.size(-1)))

        window = self._build_window(device=x_h.device, dtype=x_h.dtype)
        spec = torch.stft(
            a_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.stft_center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        spec = spec.view(B, self.analysis_dim, self.num_freq_bins, -1)

        if self.use_phase:
            feat = torch.cat([spec.real, spec.imag], dim=2)
        else:
            feat = spec.abs()

        feat = feat.permute(0, 3, 1, 2).contiguous()   # (B, n_frames, analysis_dim, num_freq_bins[*2])
        return feat.view(B, feat.size(1), -1)          # (B, n_frames, frame_desc_dim)

    def _compute_head_frame_weights(
        self,
        frame_desc: torch.Tensor,
        head_idx: int,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
    ) -> torch.Tensor:
        """
        frame_desc: (B, n_frames, frame_desc_dim)
        return: (B, n_frames)
        """
        frame_desc = self.frame_proj[head_idx](frame_desc)
        frame_score = self.frame_scorers[head_idx](frame_desc).squeeze(-1)  # (B, n_frames)

        # Optional frame-level masking derived from token mask
        frame_mask = self._build_frame_mask(
            n_frames=frame_desc.size(1),
            seq_len=seq_len,
            attention_mask=attention_mask,
            device=frame_desc.device,
        )

        if frame_mask is not None:
            frame_score = frame_score.masked_fill(~frame_mask, -1e9)

        frame_weight = torch.softmax(frame_score, dim=-1)

        if frame_mask is not None:
            frame_weight = frame_weight * frame_mask.to(dtype=frame_weight.dtype)
            frame_weight = frame_weight / frame_weight.sum(dim=-1, keepdim=True).clamp(min=self.eps)

        return frame_weight

    def _build_frame_mask(
        self,
        n_frames: int,
        seq_len: int,
        attention_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Build a boolean validity mask for STFT frames from token-level attention_mask.
        A frame is valid if it overlaps at least one valid token.
        """
        if attention_mask is None:
            return None

        B = attention_mask.size(0)
        frame_mask = torch.zeros(B, n_frames, dtype=torch.bool, device=device)

        pad_left = (self.n_fft // 2) if self.stft_center else 0

        for m in range(n_frames):
            start_pad = m * self.hop_length
            start_orig = start_pad - pad_left
            end_orig = start_orig + self.win_length
            left = max(0, start_orig)
            right = min(seq_len, end_orig)
            if right <= left:
                continue
            valid = attention_mask[:, left:right].sum(dim=-1) > 0
            frame_mask[:, m] = valid

        return frame_mask

    def _frame_weights_to_token_weights(
        self,
        frame_weight: torch.Tensor,
        seq_len: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Map per-frame scalar weights to per-token weights.

        Uniform accumulation over each frame's token support interval, then divide by coverage.
        """
        B, n_frames = frame_weight.shape
        device = frame_weight.device
        dtype = frame_weight.dtype

        token_weight = torch.zeros(B, seq_len, device=device, dtype=dtype)
        coverage = torch.zeros_like(token_weight)

        pad_left = (self.n_fft // 2) if self.stft_center else 0

        for m in range(n_frames):
            start_pad = m * self.hop_length
            start_orig = start_pad - pad_left
            end_orig = start_orig + self.win_length
            left = max(0, start_orig)
            right = min(seq_len, end_orig)
            if right <= left:
                continue
            token_weight[:, left:right] += frame_weight[:, m].unsqueeze(-1)
            coverage[:, left:right] += 1.0

        token_weight = token_weight / coverage.clamp(min=1.0)

        if attention_mask is not None:
            token_weight = token_weight * attention_mask.to(dtype=dtype)

        row_sum = token_weight.sum(dim=1, keepdim=True)
        normed = token_weight / row_sum.clamp(min=self.eps)

        if attention_mask is None:
            fallback = torch.ones(B, seq_len, device=device, dtype=dtype) / float(seq_len)
        else:
            m = attention_mask.to(dtype=dtype)
            fallback = m / m.sum(dim=1, keepdim=True).clamp(min=self.eps)

        token_weight = torch.where(row_sum >= self.eps, normed, fallback)
        return token_weight

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        """
        x: (B, T, d_model)
        attention_mask: (B, T)
        """
        B, T, _ = x.shape
        x_heads = x.view(B, T, self.num_heads, self.head_dim)

        head_outputs = []
        all_frame_weights = []
        all_token_weights = []

        for h in range(self.num_heads):
            x_h = x_heads[:, :, h, :]  # (B, T, head_dim)

            frame_desc = self._stft_frame_descriptors(x_h, self.analysis_proj[h])
            frame_weight = self._compute_head_frame_weights(
                frame_desc=frame_desc,
                head_idx=h,
                attention_mask=attention_mask,
                seq_len=T,
            )
            token_weight = self._frame_weights_to_token_weights(
                frame_weight=frame_weight,
                seq_len=T,
                attention_mask=attention_mask,
            )

            # ipdb.set_trace()
            pooled_h = (x_h * token_weight.unsqueeze(-1)).sum(dim=1)
            head_outputs.append(pooled_h)

            if return_weights:
                all_frame_weights.append(frame_weight)
                all_token_weights.append(token_weight)

        pooled = torch.cat(head_outputs, dim=-1)
        pooled = self.dropout(pooled)

        if self.gated:
            gate = torch.sigmoid(self.gate(pooled))
            pooled = pooled * gate

        pooled = self.output_proj(pooled)

        if return_weights:
            return pooled, {
                "frame_weights": all_frame_weights,   # list of length num_heads, each (B, n_frames)
                "token_weights": all_token_weights,   # list of length num_heads, each (B, T)
            }

        return pooled