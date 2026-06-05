"""
Improved Spectral Anchor Pooling with multiple optimization strategies.

This module provides enhanced versions of spectral anchor pooling that address
the limitations of the original implementation:

1. Better spectral representation (magnitude + phase vs real-only)
2. Learnable transformations (input/output projections)
3. Improved weight aggregation (soft attention vs hard max)
4. Multi-head support for richer representations

Author: Claude Code Analysis
Date: 2026-03-22
"""

import math
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llm_pooling_dropin import masked_max_pooling, masked_mean_pooling

class SpectralAnchorPoolingV2(nn.Module):
    """
    Improved spectral anchor pooling with minimal changes to the original design.

    Key improvements:
    1. Input projection layer for better feature transformation
    2. Orthogonal anchor initialization for diversity
    3. Learnable temperature for softmax scaling
    4. Soft attention aggregation instead of hard max
    5. Output projection layer

    Args:
        d_model: Hidden dimension of token features
        num_anchor: Number of learnable anchors
        use_fft: If True, compute assignment in frequency domain
        aggregation: How to aggregate anchor assignments ('soft', 'max', 'mean')
        use_projection: Whether to use input/output projections
    """

    def __init__(
        self,
        d_model: int,
        num_anchor: int = 8,
        use_fft: bool = True,
        aggregation: Literal["soft", "max", "mean"] = "soft",
        use_projection: bool = True,
    ):
        super().__init__()
        self.num_anchor = int(num_anchor)
        self.use_fft = bool(use_fft)
        self.aggregation = aggregation
        self.use_projection = use_projection
        self.d_model = d_model

        # Input transformation
        if use_projection:
            self.input_proj = nn.Linear(d_model, d_model)
            self.output_proj = nn.Linear(d_model, d_model)
        else:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()

        # Anchors with orthogonal initialization for diversity
        self.anchor = nn.Parameter(torch.empty(self.num_anchor, d_model))
        nn.init.orthogonal_(self.anchor)

        # Learnable temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))

    def _compute_spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spectral representation of input."""
        if not self.use_fft:
            return x

        # Use magnitude spectrum (more stable than real-only)
        fft_result = torch.fft.rfft(x, dim=1)
        spectral = torch.abs(fft_result)
        return spectral

    def _compute_assignment_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Compute token saliency weights based on anchor assignments."""
        # Apply input transformation
        x_transformed = self.input_proj(x)

        # Get spectral representation
        spectral = self._compute_spectral_features(x_transformed)

        # Compute scaled distances
        anchors = self.anchor.unsqueeze(0).expand(spectral.size(0), -1, -1)
        dist = torch.cdist(spectral, anchors)

        # Scale by feature dimension for numerical stability
        dist = dist / (self.d_model ** 0.5)

        # Compute soft assignment with learnable temperature
        temp = self.temperature.abs().clamp(min=0.1, max=10.0)
        alpha = torch.softmax(-dist / temp, dim=-1)

        # Aggregate assignments based on strategy
        if self.aggregation == "max":
            # Original: take max assignment confidence
            weight = alpha.max(dim=-1).values
        elif self.aggregation == "mean":
            # Average assignment weights
            weight = alpha.mean(dim=-1)
        else:  # 'soft'
            # Soft attention: weighted combination
            # Higher weight to closer anchors
            attn = torch.softmax(-dist, dim=-1)
            weight = (alpha * attn).sum(dim=-1)

        return weight

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            attention_mask: Optional mask of shape (batch, seq_len)

        Returns:
            Pooled tensor of shape (batch, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Compute assignment weights
        weight = self._compute_assignment_weights(x)

        # Interpolate back to original sequence length if FFT changed it
        if weight.size(1) != seq_len:
            weight = F.interpolate(
                weight.unsqueeze(1),
                size=seq_len,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.to(dtype=weight.dtype)
            weight = weight * mask

        # Normalize weights
        weight = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Weighted pooling
        pooled = (x * weight.view(batch_size, seq_len, 1)).sum(dim=1)

        # Output transformation
        return self.output_proj(pooled)


class MultiHeadSpectralAnchorPooling(nn.Module):
    """
    Multi-head spectral anchor pooling with learned transformations.

    This version borrows design patterns from attention pooling:
    - Multiple independent heads for diverse representations
    - Learned spectral transformations
    - Gated output mechanism

    Args:
        d_model: Hidden dimension of token features
        num_heads: Number of attention heads
        num_anchor_per_head: Number of anchors per head
        use_fft: If True, use frequency domain features
        gated: Whether to use gating mechanism
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_anchor_per_head: int = 4,
        use_fft: bool = True,
        gated: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_anchor_per_head = num_anchor_per_head
        self.use_fft = use_fft
        self.gated = gated

        # Per-head anchors
        self.anchors = nn.Parameter(
            torch.empty(num_heads, num_anchor_per_head, self.head_dim)
        )
        nn.init.orthogonal_(self.anchors)

        # Spectral transformation
        if use_fft:
            # Process both real and imaginary parts
            self.spectral_transform = nn.Linear(d_model * 2, d_model)

        # Per-head temperature
        self.temperatures = nn.Parameter(torch.ones(num_heads))

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

        # Gating mechanism
        if gated:
            self.gate = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _compute_head_attention(
        self, x_head: torch.Tensor, head_idx: int
    ) -> torch.Tensor:
        """Compute attention weights for a single head."""
        anchors = self.anchors[head_idx].unsqueeze(0).expand(x_head.size(0), -1, -1)

        # Compute distances
        dist = torch.cdist(x_head, anchors)
        dist = dist / (self.head_dim ** 0.5)

        # Softmax with per-head temperature
        temp = self.temperatures[head_idx].abs().clamp(min=0.1, max=10.0)
        attn = torch.softmax(-dist / temp, dim=-1)

        return attn

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with multi-head processing."""
        batch_size, seq_len, _ = x.shape

        # Compute spectral features if needed
        if self.use_fft:
            fft_result = torch.fft.rfft(x, dim=1)
            spectral = torch.cat([fft_result.real, fft_result.imag], dim=-1)
            # Ensure we have exactly d_model * 2 features
            spectral = spectral[:, :, : self.d_model * 2]
            if spectral.size(-1) < self.d_model * 2:
                # Pad if necessary
                padding = torch.zeros(
                    batch_size,
                    spectral.size(1),
                    self.d_model * 2 - spectral.size(-1),
                    device=spectral.device,
                    dtype=spectral.dtype,
                )
                spectral = torch.cat([spectral, padding], dim=-1)
            features = self.spectral_transform(spectral)
        else:
            features = x

        # Reshape for multi-head processing
        features = features.view(batch_size, -1, self.num_heads, self.head_dim)
        feat_seq_len = features.size(1)

        # Process each head
        head_outputs = []
        for h in range(self.num_heads):
            x_h = features[:, :, h, :]  # (B, feat_seq_len, head_dim)

            # Compute attention weights
            attn = self._compute_head_attention(x_h, h)

            # Aggregate: use mean of attention-weighted combination
            weight = attn.mean(dim=-1)  # (B, feat_seq_len)

            # Handle FFT length change - interpolate back to original seq_len
            if weight.size(1) != seq_len:
                weight = F.interpolate(
                    weight.unsqueeze(1),
                    size=seq_len,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)

            # Apply attention mask
            if attention_mask is not None:
                mask = attention_mask.to(dtype=weight.dtype)
                weight = weight * mask

            # Normalize
            weight = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-6)

            # Use original x for pooling (not spectral features)
            # Reshape x for this head
            x_orig_h = x.view(batch_size, seq_len, self.num_heads, self.head_dim)[:, :, h, :]
            pooled_h = (x_orig_h * weight.unsqueeze(-1)).sum(dim=1)
            head_outputs.append(pooled_h)

        # Concatenate head outputs
        pooled = torch.cat(head_outputs, dim=-1)  # (B, d_model)

        # Apply dropout
        pooled = self.dropout(pooled)

        # Gating
        if self.gated:
            gate = torch.sigmoid(self.gate(pooled))
            pooled = pooled * gate

        # Output projection
        return self.output_proj(pooled)


class MultiHeadLocalSpectralAnchorPooling(nn.Module):
    """
    Multi-head **local** spectral anchor pooling for 1D token sequences (e.g. AMP PLM hidden states).

    Semantics: learnable anchors live in a **per-frame local time–frequency descriptor** space
    (STFT on a small per-head analysis projection), not in a global rFFT bin axis. Frame importance
    is mapped back to residue/token positions via overlap-aware accumulation (no linear interpolate
    from frequency length to sequence length).

    **Frame-to-token mapping:** each STFT frame contributes uniformly over its supported token index
    range; contributions are averaged by per-token coverage count. This is **not** ISTFT nor
    Hann-weighted strict overlap-add reconstruction.

    **stft_center=True:** mapping uses a heuristic ``pad_left = n_fft // 2``; it may not match
    ``torch.stft`` internal padding/frame alignment exactly across PyTorch versions.

    This is **not** an MSA/evolutionary anchor replacement: it summarizes **single-sequence** local
    motif-like structure only.

    Pipeline per head:
        x_h -> analysis_proj -> STFT -> frame descriptors -> anchor softmax -> frame weights
        -> overlap-aware token weights -> masked normalize -> weighted sum of original x_h.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_anchor_per_head: int = 4,
        analysis_dim: int = 8,
        n_fft: int = 8,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        stft_center: bool = False,
        use_phase: bool = False,
        gated: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_anchor_per_head = num_anchor_per_head
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
            [nn.Linear(self.head_dim, analysis_dim, bias=True) for _ in range(num_heads)]
        )

        self.anchors = nn.Parameter(
            torch.empty(num_heads, num_anchor_per_head, self.frame_desc_dim)
        )
        nn.init.orthogonal_(self.anchors)

        self.frame_proj = nn.ModuleList(
            [
                nn.Linear(self.frame_desc_dim, self.frame_desc_dim, bias=True)
                for _ in range(num_heads)
            ]
        )

        self.temperatures = nn.Parameter(torch.ones(num_heads))

        if gated:
            self.gate = nn.Linear(d_model, d_model)

        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _build_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.hann_window(self.win_length, device=device, dtype=dtype)

    def _stft_frame_descriptors(
        self, x_h: torch.Tensor, proj: nn.Linear
    ) -> torch.Tensor:
        """x_h: (B, T, head_dim) -> frame_desc: (B, n_frames, frame_desc_dim)."""
        B, T, _ = x_h.shape
        a = proj(x_h)
        a = a.transpose(1, 2)
        a_flat = a.reshape(B * self.analysis_dim, T)

        # PyTorch STFT requires n_fft < length(signal). Pad short sequences (e.g. T<=n_fft)
        # so AMP edge cases and classification short texts do not crash; pooling still uses
        # the original seq_len T when mapping frame weights back to tokens.
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

        feat = feat.permute(0, 3, 1, 2).contiguous()
        return feat.view(B, feat.size(1), -1)

    def _compute_head_attention(
        self, frame_desc: torch.Tensor, head_idx: int
    ) -> torch.Tensor:
        frame_desc = self.frame_proj[head_idx](frame_desc)
        anchors = self.anchors[head_idx].unsqueeze(0).expand(frame_desc.size(0), -1, -1)
        dist = torch.cdist(frame_desc, anchors) / math.sqrt(frame_desc.size(-1))
        temp = self.temperatures[head_idx].abs().clamp(min=0.1, max=10.0)
        # ipdb.set_trace()
        return torch.softmax(-dist / temp, dim=-1)

    def _frame_weights_to_token_weights(
        self,
        frame_weight: torch.Tensor,
        seq_len: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Map per-frame scalar weights to per-token weights.

        Uniform accumulation over each frame's token support interval, then divide by coverage.
        Not inverse STFT; see class docstring.
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
            m = attention_mask.to(dtype=dtype).float()
            fallback = m / m.sum(dim=1, keepdim=True).clamp(min=self.eps)
        token_weight = torch.where(row_sum >= self.eps, normed, fallback)
        return token_weight

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        x_heads = x.view(B, T, self.num_heads, self.head_dim)

        head_outputs = []
        for h in range(self.num_heads):
            x_h = x_heads[:, :, h, :]
            frame_desc = self._stft_frame_descriptors(x_h, self.analysis_proj[h])
            attn = self._compute_head_attention(frame_desc, h)
            frame_weight = attn.mean(dim=-1)
            # frame_weight = attn.max(dim=-1).values
            token_weight = self._frame_weights_to_token_weights(
                frame_weight=frame_weight,
                seq_len=T,
                attention_mask=attention_mask,
            )
            # ipdb.set_trace()
            # if attention_mask is not None:
            #     m = attention_mask.float()
            #     token_weight = m / m.sum(dim=1, keepdim=True).clamp(min=self.eps)
            # else:
            #     token_weight = torch.ones(B, T, device=x_h.device, dtype=x_h.dtype) / float(T)
            pooled_h = (x_h * token_weight.unsqueeze(-1)).sum(dim=1)
            head_outputs.append(pooled_h)

        pooled = torch.cat(head_outputs, dim=-1)
        pooled = self.dropout(pooled)

        if self.gated:
            gate = torch.sigmoid(self.gate(pooled))
            pooled = pooled * gate

        return self.output_proj(pooled)


class STFTLatentAttentionMaxPooling(nn.Module):
    """
    Sequence -> STFT -> latent attention in spectral-frame space
             -> [optional ISTFT to time gate]
             -> token gating -> masked max pooling

    Modes
    -----
    return_time_gate = True:
        STFT -> latent attention -> ISTFT -> time-domain gate -> gated max pooling

    return_time_gate = False:
        STFT -> latent attention -> frame scores -> frame-to-token mapping -> token gate -> gated max pooling

    Input:
        features: (B, T, d_model)
        attention_mask: (B, T), 1 for valid token, 0 for pad

    Output:
        pooled: (B, d_model)
    """

    def __init__(
        self,
        d_model: int,
        analysis_dim: int = 16,
        num_latents: int = 8,
        num_heads: int = 4,
        n_fft: int = 8,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        dropout: float = 0.1,
        stft_center: bool = False,
        use_complex_residual: bool = True,
        return_time_gate: bool = False,
        token_gate_mode: str = "scalar",   # "scalar" or "vector"
        eps: float = 1e-6,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        if token_gate_mode not in {"scalar", "vector"}:
            raise ValueError(f"token_gate_mode must be 'scalar' or 'vector', got {token_gate_mode}")

        self.d_model = d_model
        self.analysis_dim = analysis_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else max(1, self.win_length // 2)
        self.stft_center = bool(stft_center)
        self.use_complex_residual = bool(use_complex_residual)
        self.return_time_gate = bool(return_time_gate)
        self.token_gate_mode = token_gate_mode
        self.eps = eps

        if not (1 <= self.win_length <= self.n_fft):
            raise ValueError(
                f"win_length must satisfy 1 <= win_length <= n_fft, got "
                f"win_length={self.win_length}, n_fft={self.n_fft}"
            )
        if self.hop_length < 1:
            raise ValueError(f"hop_length must be >= 1, got {self.hop_length}")

        self.num_freq_bins = self.n_fft // 2 + 1
        self.desc_dim = self.analysis_dim * self.num_freq_bins * 2  # real + imag

        # token hidden -> analysis channels
        self.analysis_proj = nn.Linear(d_model, analysis_dim)

        # trainable latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, self.desc_dim) * 0.02)

        # latent attends over frame descriptors
        self.latent_attn = nn.MultiheadAttention(
            embed_dim=self.desc_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # frame attends back to latent memory
        self.frame_attn = nn.MultiheadAttention(
            embed_dim=self.desc_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_latent = nn.LayerNorm(self.desc_dim)
        self.norm_frame = nn.LayerNorm(self.desc_dim)
        self.norm_out = nn.LayerNorm(self.desc_dim)

        self.spec_ffn = nn.Sequential(
            nn.Linear(self.desc_dim, self.desc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.desc_dim, self.desc_dim),
        )

        # for no-ISTFT branch: frame descriptor -> frame score
        self.frame_score = nn.Sequential(
            nn.Linear(self.desc_dim, self.desc_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.desc_dim // 2, 1),
        )

        # for ISTFT branch: reconstructed analysis signal -> token gate
        gate_out_dim = 1 if token_gate_mode == "scalar" else d_model
        self.time_gate = nn.Sequential(
            nn.Linear(analysis_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, gate_out_dim),
        )

        # for no-ISTFT branch: token scalar score -> token gate
        if token_gate_mode == "vector":
            self.token_expand = nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
        else:
            self.token_expand = None

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _build_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.hann_window(self.win_length, device=device, dtype=dtype)

    def _stft(self, features: torch.Tensor):
        """
        features: (B, T, d_model)
        returns:
            spec: (B, A, F, M) complex
            frame_desc: (B, M, A*F*2)
            seq_len: original T
        """
        B, T, _ = features.shape
        x = self.analysis_proj(features)              # (B, T, A)
        x = x.transpose(1, 2).contiguous()            # (B, A, T)
        x = x.view(B * self.analysis_dim, T)          # (B*A, T)

        min_len = self.n_fft + 1
        if x.size(-1) < min_len:
            x = F.pad(x, (0, min_len - x.size(-1)))

        window = self._build_window(features.device, features.dtype)

        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.stft_center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )  # (B*A, F, M)

        spec = spec.view(B, self.analysis_dim, self.num_freq_bins, -1)  # (B, A, F, M)

        spec_ri = torch.view_as_real(spec)  # (B, A, F, M, 2)
        spec_ri = spec_ri.permute(0, 3, 1, 2, 4).contiguous()  # (B, M, A, F, 2)
        frame_desc = spec_ri.view(B, spec_ri.size(1), -1)  # (B, M, D)

        return spec, frame_desc, T

    def _latent_spectral_enhance(self, frame_desc: torch.Tensor) -> torch.Tensor:
        """
        frame_desc: (B, M, D)
        return:
            enhanced_frame_desc: (B, M, D)
        """
        B = frame_desc.size(0)

        latent_queries = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, L, D)

        latent_memory, _ = self.latent_attn(
            query=latent_queries,
            key=frame_desc,
            value=frame_desc,
            need_weights=False,
        )
        latent_memory = self.norm_latent(latent_queries + self.dropout(latent_memory))

        frame_context, _ = self.frame_attn(
            query=frame_desc,
            key=latent_memory,
            value=latent_memory,
            need_weights=False,
        )
        enhanced = self.norm_frame(frame_desc + self.dropout(frame_context))
        enhanced = self.norm_out(enhanced + self.dropout(self.spec_ffn(enhanced)))

        return enhanced

    def _istft_to_token_gate(
        self,
        enhanced_frame_desc: torch.Tensor,
        original_spec: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        enhanced_frame_desc: (B, M, D)
        original_spec: (B, A, F, M) complex
        returns:
            token_gate: (B, T, 1) or (B, T, d_model)
        """
        B, M, _ = enhanced_frame_desc.shape

        enh = enhanced_frame_desc.view(B, M, self.analysis_dim, self.num_freq_bins, 2)
        enh = enh.permute(0, 2, 3, 1, 4).contiguous()  # (B, A, F, M, 2)
        enh_complex = torch.view_as_complex(enh)  # (B, A, F, M)

        if self.use_complex_residual:
            new_spec = original_spec + enh_complex
        else:
            new_spec = enh_complex

        new_spec = new_spec.view(B * self.analysis_dim, self.num_freq_bins, M)

        window = self._build_window(original_spec.device, original_spec.real.dtype)

        time_sig = torch.istft(
            new_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.stft_center,
            normalized=False,
            onesided=True,
            length=seq_len,
            return_complex=False,
        )  # (B*A, T)

        time_sig = time_sig.view(B, self.analysis_dim, seq_len).transpose(1, 2).contiguous()  # (B, T, A)

        gate = torch.sigmoid(self.time_gate(time_sig))
        return gate

    def _frame_weights_to_token_weights(
        self,
        frame_weight: torch.Tensor,
        seq_len: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        frame_weight: (B, M)
        returns:
            token_weight: (B, T)
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

        return token_weight

    def _spectral_to_token_gate(
        self,
        enhanced_frame_desc: torch.Tensor,
        seq_len: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        enhanced_frame_desc: (B, M, D)
        returns:
            token_gate: (B, T, 1) or (B, T, d_model)
        """
        frame_score = self.frame_score(enhanced_frame_desc).squeeze(-1)  # (B, M)
        frame_weight = torch.softmax(frame_score, dim=1)  # (B, M)

        token_weight = self._frame_weights_to_token_weights(
            frame_weight=frame_weight,
            seq_len=seq_len,
            attention_mask=attention_mask,
        )  # (B, T)

        if attention_mask is not None:
            token_weight = token_weight * attention_mask.to(dtype=token_weight.dtype)

        if self.token_gate_mode == "scalar":
            gate = token_weight.unsqueeze(-1)  # (B, T, 1)
        else:
            gate = torch.sigmoid(self.token_expand(token_weight.unsqueeze(-1)))  # (B, T, d_model)

        return gate

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        features: (B, T, d_model)
        attention_mask: (B, T)
        """
        spec, frame_desc, seq_len = self._stft(features)
        enhanced_frame_desc = self._latent_spectral_enhance(frame_desc)

        if self.return_time_gate:
            token_gate = self._istft_to_token_gate(
                enhanced_frame_desc=enhanced_frame_desc,
                original_spec=spec,
                seq_len=seq_len,
            )
        else:
            token_gate = self._spectral_to_token_gate(
                enhanced_frame_desc=enhanced_frame_desc,
                seq_len=seq_len,
                attention_mask=attention_mask,
            )

        gated_features = features * token_gate

        if attention_mask is not None:
            gated_features = gated_features * attention_mask.unsqueeze(-1).to(gated_features.dtype)

        pooled = masked_max_pooling(gated_features, attention_mask)
        pooled = self.out_proj(self.dropout(pooled))
        return pooled

class FFTLatentAttentionGatePooling(nn.Module):
    """
    FFT latent-attention pooling with gate-based frequency modulation.

    Pipeline
    --------
    features (B, T, D)
      -> rFFT over sequence axis
      -> frequency tokens (B, F, 2D) using [real, imag]
      -> latent attention in frequency space
      -> latent summary
      -> gate from latent summary
      -> gate-modulated frequency tokens
      -> iFFT back to time domain
      -> masked max/mean pooling
      -> output projection

    Notes
    -----
    1. No analysis_dim.
    2. No STFT.
    3. Gate is channel-wise in frequency representation.
    4. Same gate is shared across frequency bins. It is stronger than constant additive bias,
       but it is not frequency-bin-specific.
    """

    def __init__(
        self,
        d_model: int,
        num_latents: int = 16,
        num_heads: int = 8,
        dropout: float = 0.1,
        time_pool: str = "max",   # "max" or "mean"
        gate_residual: bool = True,
        eps: float = 1e-6,
        use_gate: bool = True,
        use_latent: bool = True,
    ):
        super().__init__()

        freq_dim = d_model * 2
        if freq_dim % num_heads != 0:
            raise ValueError(
                f"2*d_model={freq_dim} must be divisible by num_heads={num_heads}"
            )
        if time_pool not in {"mean", "max", "attn"}:
            raise ValueError(f"time_pool must be 'mean', 'max', or 'attn', got {time_pool}")

        self.d_model = d_model
        self.freq_dim = freq_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.time_pool = time_pool
        self.gate_residual = bool(gate_residual)
        self.use_gate = bool(use_gate)
        self.use_latent = bool(use_latent)
        self.eps = float(eps)

        # time-dimension attention pooling
        if time_pool == "attn":
            self.time_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.time_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=max(1, min(num_heads, d_model)),
                dropout=dropout,
                batch_first=True,
            )
            self.time_norm = nn.LayerNorm(d_model)

        # learnable latent queries in frequency-token space
        if self.use_latent:
            self.latents = nn.Parameter(torch.randn(num_latents, freq_dim) * 0.02)

            self.attn = nn.MultiheadAttention(
                embed_dim=freq_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

            self.norm1 = nn.LayerNorm(freq_dim)
            self.ffn = nn.Sequential(
                nn.Linear(freq_dim, freq_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(freq_dim, freq_dim),
            )
            self.norm2 = nn.LayerNorm(freq_dim)

        # latent summary -> gate
        if self.use_gate:
            if self.use_latent:
                self.freq_gate = nn.Sequential(
                    nn.Linear(freq_dim, freq_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(freq_dim, freq_dim),
                )
            else:
                # FFT-GATE: generate gate directly from freq tokens without latent attention
                self.freq_gate = nn.Sequential(
                    nn.Linear(freq_dim, freq_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(freq_dim, freq_dim),
                )

        self.time_out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _to_frequency_tokens(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        features: (B, T, D)
        returns:  freq_tokens: (B, F, 2D)
        """
        x = features
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)

        spec = torch.fft.rfft(x, dim=1)  # (B, F, D), complex
        freq_tokens = torch.cat([spec.real, spec.imag], dim=-1)  # (B, F, 2D)
        return freq_tokens

    def _latent_pool_in_frequency(self, freq_tokens: torch.Tensor) -> torch.Tensor:
        """
        freq_tokens: (B, F, 2D)
        returns: latent_out: (B, L, 2D)
        """
        B = freq_tokens.size(0)
        queries = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, L, 2D)

        attn_out, attn_weights = self.attn(
            query=queries,
            key=freq_tokens,
            value=freq_tokens,
            need_weights=True,
        )
        if attn_weights.dim() == 4:
            attn_weights = attn_weights.mean(dim=1)

        latent_out = self.norm1(queries + self.dropout(attn_out))
        latent_out = self.norm2(latent_out + self.dropout(self.ffn(latent_out)))

        self._last_latent_attn_weights = attn_weights.detach()
        self._last_latent_out = latent_out.detach()
        self._last_latent_summary = latent_out.mean(dim=1).detach()

        return latent_out

    def _apply_gate(
        self,
        freq_tokens: torch.Tensor,
        latent_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        freq_tokens: (B, F, 2D)
        latent_out:  (B, L, 2D) or None (when use_latent=False)
        returns:     (B, F, 2D)
        """
        if not self.use_gate:
            # FFT-LATENT ablation: skip gate entirely
            return freq_tokens

        if self.use_latent and latent_out is not None:
            gate_input = latent_out.mean(dim=1)  # (B, 2D)
        else:
            # FFT-GATE ablation: gate from freq token statistics
            gate_input = freq_tokens.mean(dim=1)  # (B, 2D)

        gate = torch.sigmoid(self.freq_gate(gate_input))  # (B, 2D)

        # Store raw gate values (before residual) for analysis
        self._last_raw_gate = gate.detach()

        if self.gate_residual:
            # X' = X * (1 + g)
            gate = 1.0 + gate

        enhanced_freq = freq_tokens * gate.unsqueeze(1)  # (B, F, 2D)

        # Store freq_tokens for analysis (effective weight = gate * freq_tokens)
        self._last_freq_tokens = freq_tokens.detach()
        self._last_enhanced_freq = enhanced_freq.detach()

        return enhanced_freq

    def _back_to_time(
        self,
        enhanced_freq: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        enhanced_freq: (B, F, 2D)
        returns:       (B, T, D)
        """
        real, imag = enhanced_freq.chunk(2, dim=-1)   # each: (B, F, D)
        spec = torch.complex(real, imag)              # (B, F, D)
        time_tokens = torch.fft.irfft(spec, n=seq_len, dim=1)  # (B, T, D)
        return time_tokens

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        features: (B, T, D)
        attention_mask: (B, T)
        returns: (B, D)
        """
        B, T, D = features.shape
        if D != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {D}")

        freq_tokens = self._to_frequency_tokens(features, attention_mask=attention_mask)  # (B, F, 2D)
        self._last_seq_len = T

        if self.use_latent:
            latent_out = self._latent_pool_in_frequency(freq_tokens)                      # (B, L, 2D)
            enhanced_freq = self._apply_gate(freq_tokens, latent_out)                      # (B, F, 2D)
        else:
            # FFT-GATE ablation: skip latent attention, gate from freq tokens directly
            enhanced_freq = self._apply_gate(freq_tokens, latent_out=None)                 # (B, F, 2D)

        time_tokens = self._back_to_time(enhanced_freq, seq_len=T)                        # (B, T, D)

        if self.time_pool == "mean":
            pooled = masked_mean_pooling(time_tokens, attention_mask, eps=self.eps)
        elif self.time_pool == "attn":
            B = time_tokens.shape[0]
            query = self.time_query.expand(B, -1, -1)
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = (attention_mask == 0)
            attn_out, _ = self.time_attn(query, time_tokens, time_tokens, key_padding_mask=key_padding_mask)
            pooled = self.time_norm(attn_out).squeeze(1)
        else:
            pooled = masked_max_pooling(time_tokens, attention_mask)

        pooled = self.time_out_proj(self.dropout(pooled))
        return pooled


class SEPooling(nn.Module):
    """
    Mean pooling with channel-wise gating and output projection for 1D token sequences.

    Pipeline:
        token embeddings -> masked mean pooling -> dropout
        -> sigmoid gate on pooled vector -> feature-wise rescaling
        -> output projection

    This module does not perform token-level attention, spectral analysis, or anchor-based weighting.
    It summarizes the sequence by uniform averaging over valid tokens, then applies
    channel-wise recalibration on the pooled representation.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        
        self.d_model = d_model
 
        self.eps = eps

        self.gate = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    
    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, _ = x.shape

        if attention_mask is not None:
            m = attention_mask.float()
            weight = m / m.sum(dim=1, keepdim=True).clamp(min=self.eps)
        else:
            weight = torch.ones(B, T, device=x.device, dtype=x.dtype) / float(T)
        pooled = (x * weight.unsqueeze(-1)).sum(dim=1)
        pooled = self.dropout(pooled)

        gate = torch.sigmoid(self.gate(pooled))
        pooled = pooled * gate

        return self.output_proj(pooled)


class FrequencyAwarePooling(nn.Module):
    """
    Frequency-aware pooling that directly learns frequency importance.

    Instead of using anchors, this version learns which frequency components
    are most important for the task and uses them to weight temporal features.

    Args:
        d_model: Hidden dimension
        num_freq_components: Number of frequency components to model
        combine_temporal: Whether to combine with temporal attention
    """

    def __init__(
        self,
        d_model: int,
        num_freq_components: int = 16,
        combine_temporal: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_freq_components = num_freq_components
        self.combine_temporal = combine_temporal

        # Learnable frequency importance
        self.freq_importance = nn.Parameter(torch.ones(num_freq_components))

        # Frequency to temporal mapping
        self.freq_to_temporal = nn.Linear(num_freq_components, d_model)

        # Temporal attention (if combining)
        if combine_temporal:
            self.temporal_attn = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
            )
            self.combine_gate = nn.Linear(d_model * 2, d_model)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass combining frequency and temporal information."""
        batch_size, seq_len, _ = x.shape

        # FFT
        fft_result = torch.fft.rfft(x, dim=1)
        magnitude = torch.abs(fft_result)  # (B, T//2+1, D)

        # Truncate or pad to num_freq_components
        freq_len = magnitude.size(1)
        if freq_len > self.num_freq_components:
            freq_features = magnitude[:, : self.num_freq_components, :]
        else:
            padding = torch.zeros(
                batch_size,
                self.num_freq_components - freq_len,
                self.d_model,
                device=magnitude.device,
                dtype=magnitude.dtype,
            )
            freq_features = torch.cat([magnitude, padding], dim=1)

        # Apply learned frequency importance
        freq_weights = torch.softmax(self.freq_importance, dim=0)
        freq_features = freq_features * freq_weights.view(1, -1, 1)

        # Aggregate frequency features
        freq_pooled = freq_features.mean(dim=1)  # (B, D)
        freq_pooled = self.freq_to_temporal(
            freq_features.transpose(1, 2).reshape(batch_size, self.d_model, -1)
        ).mean(dim=-1)

        if not self.combine_temporal:
            return freq_pooled

        # Temporal attention
        temporal_weights = self.temporal_attn(x).squeeze(-1)  # (B, T)

        if attention_mask is not None:
            temporal_weights = temporal_weights.masked_fill(
                ~attention_mask.bool(), float("-inf")
            )

        temporal_weights = torch.softmax(temporal_weights, dim=1)
        temporal_pooled = (x * temporal_weights.unsqueeze(-1)).sum(dim=1)

        # Combine with gating
        combined = torch.cat([freq_pooled, temporal_pooled], dim=-1)
        gate = torch.sigmoid(self.combine_gate(combined))

        return temporal_pooled + gate * freq_pooled
