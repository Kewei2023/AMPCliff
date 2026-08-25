"""FLaG core pooling: FFT latent attention gate (release-only module)."""

from typing import Optional

import torch
import torch.nn as nn

from .llm_pooling_dropin import masked_max_pooling, masked_mean_pooling


class FFTLatentAttentionGatePooling(nn.Module):
    """
    FFT latent-attention pooling with gate-based frequency modulation.

    Pipeline: rFFT -> latent attention -> gate -> iFFT -> time pooling -> projection.
    """

    def __init__(
        self,
        d_model: int,
        num_latents: int = 16,
        num_heads: int = 8,
        dropout: float = 0.1,
        time_pool: str = "max",
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
        if time_pool not in {"mean", "max"}:
            raise ValueError(f"time_pool must be 'mean' or 'max', got {time_pool}")

        self.d_model = d_model
        self.freq_dim = freq_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.time_pool = time_pool
        self.gate_residual = bool(gate_residual)
        self.use_gate = bool(use_gate)
        self.use_latent = bool(use_latent)
        self.eps = float(eps)

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

        if self.use_gate:
            self.freq_gate = nn.Sequential(
                nn.Linear(freq_dim, freq_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(freq_dim, freq_dim),
            )

        self.time_out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        if self.use_latent and not self.use_gate:
            raise ValueError(
                "use_latent=True requires use_gate=True; "
                "fft_latent_only ablation has been removed."
            )

    def _to_frequency_tokens(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = features
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        spec = torch.fft.rfft(x, dim=1)
        return torch.cat([spec.real, spec.imag], dim=-1)

    def _latent_pool_in_frequency(self, freq_tokens: torch.Tensor) -> torch.Tensor:
        B = freq_tokens.size(0)
        queries = self.latents.unsqueeze(0).expand(B, -1, -1)
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
        if not self.use_gate:
            return freq_tokens
        if self.use_latent and latent_out is not None:
            gate_input = latent_out.mean(dim=1)
        else:
            gate_input = freq_tokens.mean(dim=1)
        gate = torch.sigmoid(self.freq_gate(gate_input))
        self._last_raw_gate = gate.detach()
        if self.gate_residual:
            gate = 1.0 + gate
        enhanced_freq = freq_tokens * gate.unsqueeze(1)
        self._last_freq_tokens = freq_tokens.detach()
        self._last_enhanced_freq = enhanced_freq.detach()
        return enhanced_freq

    def _back_to_time(self, enhanced_freq: torch.Tensor, seq_len: int) -> torch.Tensor:
        real, imag = enhanced_freq.chunk(2, dim=-1)
        spec = torch.complex(real, imag)
        return torch.fft.irfft(spec, n=seq_len, dim=1)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_pre_projection: bool = False,
    ):
        _, T, D = features.shape
        if D != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {D}")

        freq_tokens = self._to_frequency_tokens(features, attention_mask=attention_mask)
        self._last_seq_len = T

        if self.use_latent:
            latent_out = self._latent_pool_in_frequency(freq_tokens)
            enhanced_freq = self._apply_gate(freq_tokens, latent_out)
        else:
            enhanced_freq = self._apply_gate(freq_tokens, latent_out=None)

        time_tokens = self._back_to_time(enhanced_freq, seq_len=T)

        if self.time_pool == "mean":
            pooled = masked_mean_pooling(time_tokens, attention_mask, eps=self.eps)
        else:
            pooled = masked_max_pooling(time_tokens, attention_mask)

        pooled_pre_projection = pooled
        pooled_output = self.time_out_proj(self.dropout(pooled_pre_projection))
        if return_pre_projection:
            return pooled_output, pooled_pre_projection
        return pooled_output
