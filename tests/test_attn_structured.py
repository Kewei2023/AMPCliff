import sys
from pathlib import Path

import pytest
import torch

REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from AMPCliff.factory.pooling.llm_pooling_dropin import StructuredSelfAttentivePooling
from AMPCliff.utils import attention_penalty as penalty_mod


def _official_frobenius(mat: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    size = mat.size()
    if len(size) != 3:
        raise ValueError("matrix for computing Frobenius norm should be with 3 dims")
    ret = (mat ** 2).sum(dim=(1, 2)) + eps
    return ret.sqrt().sum() / size[0]


def test_attn_structured_output_shape():
    torch.manual_seed(0)
    B, T, D = 2, 13, 64
    hops = 4
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T)
    mask[:, -3:] = 0

    pooler = StructuredSelfAttentivePooling(
        hidden_size=D,
        attention_size=16,
        attention_hops=hops,
        dropout=0.0,
        use_bias=False,
    )
    out = pooler(x, mask)
    assert out.shape == (B, D)
    assert torch.isfinite(out).all()


def test_attn_structured_attention_weights_shape_and_normalization():
    torch.manual_seed(0)
    B, T, D = 2, 10, 32
    hops = 3
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T)
    mask[0, -2:] = 0
    mask[1, -4:] = 0

    pooler = StructuredSelfAttentivePooling(
        hidden_size=D,
        attention_size=8,
        attention_hops=hops,
        dropout=0.0,
    )
    pooler(x, mask)
    weights = pooler._last_attention_weights
    assert weights is not None
    assert weights.shape == (B, hops, T)
    for b in range(B):
        valid = mask[b].bool()
        for h in range(hops):
            assert torch.allclose(weights[b, h, valid].sum(), torch.tensor(1.0), atol=1e-5)
            assert torch.allclose(weights[b, h, ~valid], torch.zeros_like(weights[b, h, ~valid]))


def test_attn_structured_penalty_and_gradients():
    torch.manual_seed(0)
    B, T, D = 2, 8, 32
    hops = 3
    x = torch.randn(B, T, D, requires_grad=True)
    mask = torch.ones(B, T)

    pooler = StructuredSelfAttentivePooling(
        hidden_size=D,
        attention_size=8,
        attention_hops=hops,
        dropout=0.0,
        use_bias=False,
    )
    out = pooler(x, mask)
    penalty = pooler.compute_penalty_loss(coeff=1.0)
    assert penalty.ndim == 0
    (out.sum() + penalty).backward()
    assert pooler.ws1.weight.grad is not None
    assert pooler.ws2.weight.grad is not None
    assert torch.linalg.norm(pooler.ws1.weight.grad) > 0
    assert torch.linalg.norm(pooler.ws2.weight.grad) > 0


def test_frobenius_matches_official():
    mat = torch.randn(3, 4, 4)
    official = _official_frobenius(mat)
    ours = penalty_mod.frobenius_norm_batched(mat)
    assert torch.allclose(official, ours, atol=1e-6)


def test_structured_attention_penalty_matches_official_flow():
    torch.manual_seed(0)
    attention = torch.softmax(torch.randn(2, 3, 7), dim=-1)
    attention_t = attention.transpose(1, 2).contiguous()
    gram = torch.bmm(attention, attention_t)
    eye = torch.eye(gram.size(-1), device=gram.device, dtype=gram.dtype)
    diff = gram - eye.unsqueeze(0)[: attention.size(0)]
    expected = _official_frobenius(diff)
    actual = penalty_mod.structured_attention_penalty(attention)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_attn_structured_no_bias_and_out_proj_shape():
    D = 32
    hops = 5
    pooler = StructuredSelfAttentivePooling(
        hidden_size=D,
        attention_size=8,
        attention_hops=hops,
        use_bias=False,
        hop_output="flatten",
    )
    assert pooler.out_proj.in_features == hops * D
    assert pooler.ws1.bias is None
    assert pooler.ws2.bias is None
    assert pooler.out_proj.bias is None


def test_attn_structured_invalid_hop_output_raises():
    with pytest.raises(ValueError, match="hop_output"):
        StructuredSelfAttentivePooling(hidden_size=32, hop_output="mean")
