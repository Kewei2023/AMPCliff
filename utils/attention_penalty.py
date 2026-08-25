"""Frobenius orthogonality penalty for structured self-attention (Lin et al.)."""

from __future__ import annotations

import torch


def frobenius_norm_batched(mat: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Frobenius norm per batch matrix, averaged over batch."""
    if mat.ndim != 3:
        raise ValueError("matrix for computing Frobenius norm should be with 3 dims")
    ret = (mat ** 2).sum(dim=(1, 2)) + eps
    return ret.sqrt().sum() / mat.size(0)


def structured_attention_penalty(
    attention: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute ||A A^T - I||_F averaged over batch."""
    attention_t = attention.transpose(1, 2).contiguous()
    gram = torch.bmm(attention, attention_t)
    eye = torch.eye(gram.size(-1), device=gram.device, dtype=gram.dtype)
    diff = gram - eye.unsqueeze(0)[: attention.size(0)]
    return frobenius_norm_batched(diff, eps=eps)
