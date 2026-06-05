from typing import Optional

import torch

from .spectral_anchor_v2 import MultiHeadLocalSpectralAnchorPooling
import ipdb

class MultiHeadLocalSpectralAnchorPoolingNew(MultiHeadLocalSpectralAnchorPooling):
    """
    New-version local spectral anchor pooling.

    Compared with the legacy class, this variant computes frame weights using
    anchor-importance weighted assignment:
        anchor_importance = mean over frames
        frame_weight = sum(anchor_importance * assignment)
    """

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x_heads = x.view(bsz, seq_len, self.num_heads, self.head_dim)

        head_outputs = []
        for head_idx in range(self.num_heads):
            x_h = x_heads[:, :, head_idx, :]
            # frame_desc = self._stft_frame_descriptors(x_h, self.analysis_proj[head_idx])
            # attn = self._compute_head_attention(frame_desc, head_idx)
            # anchor_importance = attn.mean(dim=1)
            # frame_weight = (anchor_importance.unsqueeze(1) * attn).sum(dim=-1)
            # token_weight = self._frame_weights_to_token_weights(
            #     frame_weight=frame_weight,
            #     seq_len=seq_len,
            #     attention_mask=attention_mask,
            # )
            # ipdb.set_trace()
            if attention_mask is not None:
                m = attention_mask.float()
                token_weight = m / m.sum(dim=1, keepdim=True).clamp(min=self.eps)
            else:
                token_weight = torch.ones(bsz, seq_len, device=x_h.device, dtype=x_h.dtype) / float(seq_len)
            
            pooled_h = (x_h * token_weight.unsqueeze(-1)).sum(dim=1)
            head_outputs.append(pooled_h)

        pooled = torch.cat(head_outputs, dim=-1)
        pooled = self.dropout(pooled)

        if self.gated:
            gate = torch.sigmoid(self.gate(pooled))
            pooled = pooled * gate

        return self.output_proj(pooled)
