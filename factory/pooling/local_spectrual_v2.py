# maintained by kewei li
import math
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleAnchorFreeSTFTPoolingV2(nn.Module):
    """
    Multi-scale anchor-free local STFT pooling for 1D token sequences.

    Main ideas:
        1. Multi-scale STFT per head
        2. Per-scale frame scoring
        3. Learnable scale fusion
        4. Residual mean branch for optimization stability
        5. Temperature sharpening on frame softmax

    Input:
        x:              (B, T, d_model)
        attention_mask: (B, T), optional

    Output:
        pooled:         (B, d_model)

    Optional debug output:
        frame_weights per head / per scale
        token_weights per head
        scale_gates per head
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        analysis_dim: int = 8,
        n_ffts: Optional[List[int]] = None,
        win_lengths: Optional[List[int]] = None,
        hop_lengths: Optional[List[int]] = None,
        stft_center: bool = False,
        use_phase: bool = False,
        scorer_hidden_dim: Optional[int] = None,
        frame_temperature: float = 0.5,
        gated: bool = True,
        output_proj: bool = True,
        residual_mix_init: float = 0.1,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.analysis_dim = analysis_dim
        self.stft_center = bool(stft_center)
        self.use_phase = bool(use_phase)
        self.gated = bool(gated)
        self.use_output_proj = bool(output_proj)
        self.eps = eps

        if n_ffts is None:
            n_ffts = [4, 8]
        self.n_scales = len(n_ffts)
        if self.n_scales < 1:
            raise ValueError("At least one STFT scale is required")

        if win_lengths is None:
            win_lengths = list(n_ffts)
        if hop_lengths is None:
            hop_lengths = [max(1, w // 2) for w in win_lengths]

        if not (len(n_ffts) == len(win_lengths) == len(hop_lengths)):
            raise ValueError("n_ffts, win_lengths, hop_lengths must have the same length")

        self.n_ffts = n_ffts
        self.win_lengths = win_lengths
        self.hop_lengths = hop_lengths

        for n_fft, win_length, hop_length in zip(self.n_ffts, self.win_lengths, self.hop_lengths):
            if n_fft < 1:
                raise ValueError(f"n_fft must be >= 1, got {n_fft}")
            if not (1 <= win_length <= n_fft):
                raise ValueError(
                    f"win_length must satisfy 1 <= win_length <= n_fft, "
                    f"got win_length={win_length}, n_fft={n_fft}"
                )
            if hop_length < 1:
                raise ValueError(f"hop_length must be >= 1, got {hop_length}")

        # Per-scale descriptor dims
        self.num_freq_bins = [n_fft // 2 + 1 for n_fft in self.n_ffts]
        if self.use_phase:
            self.frame_desc_dims = [analysis_dim * fb * 2 for fb in self.num_freq_bins]
        else:
            self.frame_desc_dims = [analysis_dim * fb for fb in self.num_freq_bins]

        if scorer_hidden_dim is None:
            scorer_hidden_dim = max(32, max(self.frame_desc_dims))

        # One analysis projection per head, shared across scales
        self.analysis_proj = nn.ModuleList([
            nn.Linear(self.head_dim, self.analysis_dim, bias=True)
            for _ in range(self.num_heads)
        ])

        # Per-head, per-scale frame projection
        self.frame_proj = nn.ModuleList()
        self.frame_scorers = nn.ModuleList()
        self.frame_temperatures = nn.ParameterList()

        for _h in range(self.num_heads):
            head_frame_proj = nn.ModuleList()
            head_frame_scorers = nn.ModuleList()
            for s in range(self.n_scales):
                d_in = self.frame_desc_dims[s]
                head_frame_proj.append(
                    nn.Sequential(
                        nn.LayerNorm(d_in),
                        nn.Linear(d_in, d_in, bias=True),
                        nn.GELU(),
                    )
                )
                head_frame_scorers.append(
                    nn.Sequential(
                        nn.LayerNorm(d_in),
                        nn.Linear(d_in, scorer_hidden_dim, bias=True),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(scorer_hidden_dim, scorer_hidden_dim // 2, bias=True),
                        nn.GELU(),
                        nn.Linear(scorer_hidden_dim // 2, 1, bias=True),
                    )
                )
                self.frame_temperatures.append(
                    nn.Parameter(torch.tensor(float(frame_temperature)))
                )

            self.frame_proj.append(head_frame_proj)
            self.frame_scorers.append(head_frame_scorers)

        # Per-head scale fusion gate: use pooled per-scale descriptors -> logits over scales
        self.scale_gate_mlps = nn.ModuleList()
        for _h in range(self.num_heads):
            mlps = nn.ModuleList()
            for s in range(self.n_scales):
                d_in = self.frame_desc_dims[s]
                mlps.append(
                    nn.Sequential(
                        nn.LayerNorm(d_in),
                        nn.Linear(d_in, max(16, d_in // 2)),
                        nn.GELU(),
                    )
                )
            self.scale_gate_mlps.append(mlps)

        self.scale_gate_out = nn.ModuleList([
            nn.Linear(sum(max(16, d // 2) for d in self.frame_desc_dims), self.n_scales, bias=True)
            for _ in range(self.num_heads)
        ])

        # Residual mean mixing coefficient, learned and bounded to positive
        self.residual_mix = nn.Parameter(torch.tensor(float(residual_mix_init)))

        self.dropout = nn.Dropout(dropout)

        if self.gated:
            self.gate = nn.Linear(self.d_model, self.d_model)

        if self.use_output_proj:
            self.output_proj = nn.Linear(self.d_model, self.d_model)

    def _build_window(self, win_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.hann_window(win_length, device=device, dtype=dtype)

    def _stft_frame_descriptors(
        self,
        x_h: torch.Tensor,
        proj: nn.Linear,
        n_fft: int,
        win_length: int,
        hop_length: int,
    ) -> torch.Tensor:
        """
        x_h: (B, T, head_dim)
        return: (B, n_frames, frame_desc_dim)
        """
        B, T, _ = x_h.shape

        a = proj(x_h)              # (B, T, analysis_dim)
        a = a.transpose(1, 2)      # (B, analysis_dim, T)
        a_flat = a.reshape(B * self.analysis_dim, T)

        min_len = n_fft + 1
        if a_flat.size(-1) < min_len:
            a_flat = F.pad(a_flat, (0, min_len - a_flat.size(-1)))

        window = self._build_window(win_length=win_length, device=x_h.device, dtype=x_h.dtype)
        spec = torch.stft(
            a_flat,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=self.stft_center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        num_freq_bins = n_fft // 2 + 1
        spec = spec.view(B, self.analysis_dim, num_freq_bins, -1)

        if self.use_phase:
            feat = torch.cat([spec.real, spec.imag], dim=2)
        else:
            feat = spec.abs()

        feat = feat.permute(0, 3, 1, 2).contiguous()
        return feat.view(B, feat.size(1), -1)

    def _build_frame_mask(
        self,
        n_frames: int,
        seq_len: int,
        attention_mask: Optional[torch.Tensor],
        n_fft: int,
        win_length: int,
        hop_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None

        B = attention_mask.size(0)
        frame_mask = torch.zeros(B, n_frames, dtype=torch.bool, device=device)
        pad_left = (n_fft // 2) if self.stft_center else 0

        for m in range(n_frames):
            start_pad = m * hop_length
            start_orig = start_pad - pad_left
            end_orig = start_orig + win_length
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
        attention_mask: Optional[torch.Tensor],
        n_fft: int,
        win_length: int,
        hop_length: int,
    ) -> torch.Tensor:
        """
        Uniform overlap-aware mapping from frame weights to token weights.
        This follows the same general mapping strategy as your current local spectral pooling. :contentReference[oaicite:1]{index=1}
        """
        B, n_frames = frame_weight.shape
        device = frame_weight.device
        dtype = frame_weight.dtype

        token_weight = torch.zeros(B, seq_len, device=device, dtype=dtype)
        coverage = torch.zeros_like(token_weight)

        pad_left = (n_fft // 2) if self.stft_center else 0

        for m in range(n_frames):
            start_pad = m * hop_length
            start_orig = start_pad - pad_left
            end_orig = start_orig + win_length
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

    def _masked_mean(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        if attention_mask is None:
            return x.mean(dim=1)
        m = attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=self.eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        B, T, _ = x.shape
        x_heads = x.view(B, T, self.num_heads, self.head_dim)

        head_outputs = []

        debug_frame_weights: List[List[torch.Tensor]] = []
        debug_token_weights: List[torch.Tensor] = []
        debug_scale_gates: List[torch.Tensor] = []

        for h in range(self.num_heads):
            x_h = x_heads[:, :, h, :]   # (B, T, head_dim)

            per_scale_frame_weights = []
            per_scale_token_weights = []
            per_scale_gate_features = []

            for s in range(self.n_scales):
                n_fft = self.n_ffts[s]
                win_length = self.win_lengths[s]
                hop_length = self.hop_lengths[s]

                frame_desc = self._stft_frame_descriptors(
                    x_h=x_h,
                    proj=self.analysis_proj[h],
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                )

                frame_desc = self.frame_proj[h][s](frame_desc)

                frame_score = self.frame_scorers[h][s](frame_desc).squeeze(-1)  # (B, n_frames)

                frame_mask = self._build_frame_mask(
                    n_frames=frame_desc.size(1),
                    seq_len=T,
                    attention_mask=attention_mask,
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    device=frame_desc.device,
                )
                if frame_mask is not None:
                    frame_score = frame_score.masked_fill(~frame_mask, -1e9)

                temp = self.frame_temperatures[h * self.n_scales + s].abs().clamp(min=0.05, max=10.0)
                frame_weight = torch.softmax(frame_score / temp, dim=-1)

                if frame_mask is not None:
                    frame_weight = frame_weight * frame_mask.to(dtype=frame_weight.dtype)
                    frame_weight = frame_weight / frame_weight.sum(dim=-1, keepdim=True).clamp(min=self.eps)

                token_weight = self._frame_weights_to_token_weights(
                    frame_weight=frame_weight,
                    seq_len=T,
                    attention_mask=attention_mask,
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                )

                # For scale gating, summarize descriptor over frames
                if frame_mask is not None:
                    fm = frame_mask.unsqueeze(-1).to(dtype=frame_desc.dtype)
                    desc_summary = (frame_desc * fm).sum(dim=1) / fm.sum(dim=1).clamp(min=self.eps)
                else:
                    desc_summary = frame_desc.mean(dim=1)

                gate_feat = self.scale_gate_mlps[h][s](desc_summary)

                per_scale_frame_weights.append(frame_weight)
                per_scale_token_weights.append(token_weight)
                per_scale_gate_features.append(gate_feat)

            # Learnable scale fusion
            gate_feat_cat = torch.cat(per_scale_gate_features, dim=-1)
            scale_logits = self.scale_gate_out[h](gate_feat_cat)     # (B, n_scales)
            scale_gate = torch.softmax(scale_logits, dim=-1)         # (B, n_scales)

            fused_token_weight = 0.0
            for s in range(self.n_scales):
                fused_token_weight = fused_token_weight + scale_gate[:, s].unsqueeze(-1) * per_scale_token_weights[s]

            # normalize once more after scale fusion
            if attention_mask is not None:
                fused_token_weight = fused_token_weight * attention_mask.to(dtype=fused_token_weight.dtype)
            fused_token_weight = fused_token_weight / fused_token_weight.sum(dim=-1, keepdim=True).clamp(min=self.eps)

            pooled_stft = (x_h * fused_token_weight.unsqueeze(-1)).sum(dim=1)
            pooled_mean = self._masked_mean(x_h, attention_mask)

            mix = self.residual_mix.abs()
            pooled_h = pooled_mean + mix * pooled_stft

            head_outputs.append(pooled_h)

            if return_weights:
                debug_frame_weights.append(per_scale_frame_weights)
                debug_token_weights.append(fused_token_weight)
                debug_scale_gates.append(scale_gate)

        pooled = torch.cat(head_outputs, dim=-1)
        pooled = self.dropout(pooled)

        if self.gated:
            gate = torch.sigmoid(self.gate(pooled))
            pooled = pooled * gate

        if self.use_output_proj:
            pooled = self.output_proj(pooled)

        if return_weights:
            return pooled, {
                "frame_weights": debug_frame_weights,  # [head][scale] -> (B, n_frames)
                "token_weights": debug_token_weights,  # [head] -> (B, T)
                "scale_gates": debug_scale_gates,      # [head] -> (B, n_scales)
                "residual_mix": self.residual_mix.detach(),
            }

        return pooled