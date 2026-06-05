import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean_pooling(features: torch.Tensor, attention_mask: Optional[torch.Tensor], eps: float = 1e-9) -> torch.Tensor:
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


class VectorAttentionHead(nn.Module):
    def __init__(self, d_model: int, d_attn: int, temperature: float = 1.0, gated: bool = True):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_attn)
        self.W2 = nn.Linear(d_attn, d_attn)
        self.W3 = nn.Linear(d_attn, d_attn)
        self.temperature = float(temperature)
        self.gated = gated
        if gated:
            self.U = nn.Linear(d_model, d_attn)

    def forward(self, X: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        S = self.W1(X)
        S1 = torch.relu(self.W2(S))
        e = self.W3(S1)
        if self.gated:
            e = e * torch.sigmoid(self.U(X))
        if attention_mask is not None:
            m = attention_mask.bool().unsqueeze(-1)
            neg_inf = torch.finfo(X.dtype).min
            e = torch.where(m, e, torch.full_like(e, neg_inf))
        if self.temperature != 1.0:
            e = e / self.temperature
        A = F.softmax(e.transpose(1, 2), dim=2).transpose(1, 2)
        v = (A * S).sum(dim=1)
        return v, A


class MultiHeadVectorAttnPooling(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_attn: Optional[int] = None,
        temperature: float = 1.0,
        gated: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_attn is None:
            assert d_model % num_heads == 0
            d_attn = d_model // num_heads
        assert num_heads * d_attn == d_model
        self.heads = nn.ModuleList([
            VectorAttentionHead(d_model, d_attn, temperature=temperature, gated=gated)
            for _ in range(num_heads)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outs = []
        for head in self.heads:
            v, _ = head(X, attention_mask)
            outs.append(v)
        return self.dropout(torch.cat(outs, dim=-1))


class LatentAttentionPooling(nn.Module):
    """Trainable latent pooling similar in spirit to recent LLM embedding heads."""
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
        pooled, _ = self.attn(queries, features, features, key_padding_mask=key_padding_mask, need_weights=False)
        pooled = pooled.mean(dim=1)
        return self.out_proj(self.dropout(pooled))


class MultiLayerTrainablePooling(nn.Module):
    """Pool across multiple hidden layers and tokens."""
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.token_pool = MultiHeadVectorAttnPooling(
            d_model=hidden_size,
            num_heads=num_heads,
            temperature=1.0,
            gated=True,
            dropout=dropout,
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, all_hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # all_hidden_states: (B, T, D, L)
        layer_w = torch.softmax(self.layer_logits, dim=0)
        fused = (all_hidden_states * layer_w.view(1, 1, 1, -1)).sum(dim=-1)
        pooled = self.token_pool(fused, attention_mask)
        return self.out_proj(pooled)


class SlicedWassersteinPooling(nn.Module):
    """A lightweight distributional pooling approximation.

    It projects token embeddings to several 1D directions, computes quantiles,
    then maps the concatenated quantiles back to hidden size.
    """
    def __init__(self, d_model: int, num_projections: int = 32, num_quantiles: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_projections = num_projections
        self.num_quantiles = num_quantiles
        self.proj = nn.Parameter(torch.randn(num_projections, d_model))
        self.out = nn.Sequential(
            nn.Linear(num_projections * num_quantiles, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        proj = F.normalize(self.proj, dim=-1)
        scores = torch.einsum('btd,pd->btp', features, proj)
        if attention_mask is not None:
            mask = attention_mask.bool().unsqueeze(-1)
            scores = scores.masked_fill(~mask, float('nan'))
        qs = torch.linspace(0.0, 1.0, self.num_quantiles, device=features.device)
        # torch.nanquantile is available on modern PyTorch
        quantiles = torch.nanquantile(scores, qs, dim=1).permute(1, 2, 0).contiguous()
        quantiles = torch.nan_to_num(quantiles, nan=0.0)
        return self.out(quantiles.view(features.size(0), -1))


class MultiHeadLocalSpectralAnchorPooling(nn.Module):
    """
    Multi-head local spectral anchor pooling for 1D token sequences.
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
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_anchor_per_head = num_anchor_per_head
        self.analysis_dim = analysis_dim
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else max(1, self.win_length // 2)
        self.stft_center = stft_center
        self.use_phase = use_phase
        self.gated = gated
        self.eps = eps
        self.num_freq_bins = self.n_fft // 2 + 1
        self.frame_desc_dim = self.analysis_dim * self.num_freq_bins * (2 if self.use_phase else 1)
        self.analysis_proj = nn.ModuleList([nn.Linear(self.head_dim, analysis_dim) for _ in range(num_heads)])
        self.anchors = nn.Parameter(torch.empty(num_heads, num_anchor_per_head, self.frame_desc_dim))
        nn.init.orthogonal_(self.anchors)
        self.frame_proj = nn.ModuleList([nn.Linear(self.frame_desc_dim, self.frame_desc_dim) for _ in range(num_heads)])
        self.temperatures = nn.Parameter(torch.ones(num_heads))
        if gated:
            self.gate = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _build_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.hann_window(self.win_length, device=device, dtype=dtype)

    def _stft_frame_descriptors(self, x_h: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        B, T, _ = x_h.shape
        a = proj(x_h).transpose(1, 2)
        a_flat = a.reshape(B * self.analysis_dim, T)
        min_len = self.n_fft + 1
        if a_flat.size(-1) < min_len:
            a_flat = F.pad(a_flat, (0, min_len - a_flat.size(-1)))
        window = self._build_window(x_h.device, x_h.dtype)
        spec = torch.stft(
            a_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.stft_center,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = spec.view(B, self.analysis_dim, self.num_freq_bins, -1)
        feat = torch.cat([spec.real, spec.imag], dim=2) if self.use_phase else spec.abs()
        feat = feat.permute(0, 3, 1, 2).contiguous()
        return feat.view(B, feat.size(1), -1)

    def _compute_head_attention(self, frame_desc: torch.Tensor, head_idx: int) -> torch.Tensor:
        frame_desc = self.frame_proj[head_idx](frame_desc)
        anchors = self.anchors[head_idx].unsqueeze(0).expand(frame_desc.size(0), -1, -1)
        dist = torch.cdist(frame_desc, anchors) / math.sqrt(frame_desc.size(-1))
        temp = self.temperatures[head_idx].abs().clamp(min=0.1, max=10.0)
        return torch.softmax(-dist / temp, dim=-1)

    def _frame_weights_to_token_weights(self, frame_weight: torch.Tensor, seq_len: int, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, n_frames = frame_weight.shape
        token_weight = torch.zeros(B, seq_len, device=frame_weight.device, dtype=frame_weight.dtype)
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
            token_weight = token_weight * attention_mask.to(dtype=token_weight.dtype)
        row_sum = token_weight.sum(dim=1, keepdim=True)
        normed = token_weight / row_sum.clamp(min=self.eps)
        if attention_mask is None:
            fallback = torch.ones(B, seq_len, device=token_weight.device, dtype=token_weight.dtype) / float(seq_len)
        else:
            m = attention_mask.to(dtype=token_weight.dtype)
            fallback = m / m.sum(dim=1, keepdim=True).clamp(min=self.eps)
        return torch.where(row_sum >= self.eps, normed, fallback)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        x_heads = x.view(B, T, self.num_heads, self.head_dim)
        head_outputs = []
        for h in range(self.num_heads):
            x_h = x_heads[:, :, h, :]
            frame_desc = self._stft_frame_descriptors(x_h, self.analysis_proj[h])
            attn = self._compute_head_attention(frame_desc, h)
            frame_weight = attn.mean(dim=-1)
            token_weight = self._frame_weights_to_token_weights(frame_weight, T, attention_mask)
            pooled_h = (x_h * token_weight.unsqueeze(-1)).sum(dim=1)
            head_outputs.append(pooled_h)
        pooled = self.dropout(torch.cat(head_outputs, dim=-1))
        if self.gated:
            pooled = pooled * torch.sigmoid(self.gate(pooled))
        return self.output_proj(pooled)


class LLMFlexiblePoolingHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        dropout_prob = getattr(config, 'hidden_dropout_prob', 0.1)
        self.pooling = config.pooling
        self.mltp_pooler: Optional[MultiLayerTrainablePooling] = None

        if self.pooling == 'attn':
            self.pool = MultiHeadVectorAttnPooling(hidden_size, num_heads=8, gated=True, dropout=dropout_prob)
        elif self.pooling == 'latent_attn':
            self.pool = LatentAttentionPooling(
                hidden_size,
                num_latents=getattr(config, 'num_latents', 8),
                num_heads=8,
                dropout=dropout_prob,
            )
        elif self.pooling == 'swe_ot':
            self.pool = SlicedWassersteinPooling(
                hidden_size,
                num_projections=getattr(config, 'num_projections', 32),
                num_quantiles=getattr(config, 'num_quantiles', 8),
                dropout=dropout_prob,
            )
        elif self.pooling == 'mltp':
            # Important: create learnable parameters in __init__ so they get registered.
            num_layers = getattr(config, 'num_hidden_layers', None)
            if num_layers is None:
                raise ValueError(
                    "For pooling='mltp', config.num_hidden_layers is required to allocate layer_logits."
                )
            mltp_num_heads = getattr(config, 'mltp_num_heads', getattr(config, 'num_heads', 8))
            self.mltp_pooler = MultiLayerTrainablePooling(
                hidden_size=hidden_size,
                num_layers=int(num_layers),
                num_heads=int(mltp_num_heads),
                dropout=dropout_prob,
            )
            self.pool = None
        elif self.pooling == 'mlsap':
            self.pool = MultiHeadLocalSpectralAnchorPooling(
                hidden_size,
                num_heads=getattr(config, 'mlsap_num_heads', 4),
                num_anchor_per_head=getattr(config, 'mlsap_num_anchor_per_head', 4),
                analysis_dim=getattr(config, 'mlsap_analysis_dim', 8),
                n_fft=getattr(config, 'mlsap_n_fft', 8),
                win_length=getattr(config, 'mlsap_win_length', None),
                hop_length=getattr(config, 'mlsap_hop_length', None),
                stft_center=getattr(config, 'mlsap_stft_center', False),
                use_phase=getattr(config, 'mlsap_use_phase', False),
                gated=True,
                dropout=dropout_prob,
            )
        else:
            self.pool = None

        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, getattr(config, 'num_labels', 1))

    def _pool_last_hidden(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling == 'mean':
            return masked_mean_pooling(features, attention_mask)
        if self.pooling == 'max':
            return masked_max_pooling(features, attention_mask)
        if self.pooling == 'last':
            return last_token_pooling(features, attention_mask)
        if self.pooling in {'attn', 'latent_attn', 'swe_ot', 'mlsap'}:
            return self.pool(features, attention_mask)
        raise ValueError(f'Unsupported pooling: {self.pooling}')

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        all_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pooling == 'mltp':
            if all_hidden_states is None:
                raise ValueError('all_hidden_states is required for mltp pooling')
            if self.mltp_pooler is None:
                raise RuntimeError("mltp_pooler was not initialized; check __init__.")
            x = self.mltp_pooler(all_hidden_states, attention_mask)
        else:
            x = self._pool_last_hidden(last_hidden_state, attention_mask)
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class RegModel_FlexiblePooling(nn.Module):
    """Drop-in replacement for current RegModel_v2 under LLM backbone."""
    def __init__(self, pretrain_model: nn.Module, config):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.head = LLMFlexiblePoolingHead(config)

    def forward(self, batch1):
        outputs = self.pretrain_model(**batch1)
        attention_mask = batch1.get('attention_mask')
        features1 = outputs.last_hidden_state
        all_layer_features = torch.stack(list(outputs.hidden_states)[1:], dim=-1)
        regression_output1 = self.head(features1, attention_mask, all_layer_features)
        return regression_output1, features1, all_layer_features, None, None


__all__ = [
    # token-level pooling used by AMPCliff registry / regression wiring
    "last_token_pooling",
    "LatentAttentionPooling",
    "SlicedWassersteinPooling",
    "MultiLayerTrainablePooling",
]
