import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralAnchorPooling(nn.Module):
    """
    Frequency-aware anchor pooling for sequence features.

    Args:
        d_model: hidden dimension of token features.
        num_anchor: number of learnable anchors.
        use_fft: if True, assignment is computed in frequency domain.
    """

    def __init__(self, d_model: int, num_anchor: int = 8, use_fft: bool = True):
        super().__init__()
        self.num_anchor = int(num_anchor)
        self.use_fft = bool(use_fft)
        self.anchor = nn.Parameter(torch.empty(self.num_anchor, d_model))
        nn.init.xavier_uniform_(self.anchor)

    def _compute_assignment_weights(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        if self.use_fft:
            spectral = torch.fft.rfft(x, dim=1).real
        else:
            spectral = x

        anchors = self.anchor.unsqueeze(0).expand(spectral.size(0), -1, -1)
        dist = torch.cdist(spectral, anchors)  # (B, T_or_Tfft, K)
        alpha = torch.softmax(-dist, dim=-1)
        # Use assignment confidence as token saliency.
        weight = alpha.max(dim=-1).values  # (B, T_or_Tfft)
        return weight

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, d)
        batch_size, seq_len, _ = x.shape
        weight = self._compute_assignment_weights(x)

        # rfft changes sequence length, project weights back to token axis.
        if weight.size(1) != seq_len:
            weight = F.interpolate(
                weight.unsqueeze(1),
                size=seq_len,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        if attention_mask is not None:
            mask = attention_mask.to(dtype=weight.dtype)
            weight = weight * mask

        weight = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (x * weight.view(batch_size, seq_len, 1)).sum(dim=1)
        return pooled
