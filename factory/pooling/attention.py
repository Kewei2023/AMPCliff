# maintained by kewei li
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorAttentionHead(nn.Module):
    def __init__(self, d_model: int, d_attn: int, temperature: float = 1.0, gated: bool = True):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_attn, bias=True)
        self.W2 = nn.Linear(d_attn, d_attn, bias=True)
        self.W3 = nn.Linear(d_attn, d_attn, bias=True)
        self.temperature = float(temperature)
        self.gated = bool(gated)
        if gated:
            self.U = nn.Linear(d_model, d_attn, bias=True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.zeros_(self.W1.bias)
        if self.gated:
            nn.init.xavier_uniform_(self.U.weight)
            nn.init.zeros_(self.U.bias)

        nn.init.kaiming_uniform_(self.W2.weight, nonlinearity="relu")
        nn.init.zeros_(self.W2.bias)
        nn.init.kaiming_uniform_(self.W3.weight, nonlinearity="relu")
        nn.init.zeros_(self.W3.bias)

    def forward(self, X: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        S = self.W1(X)
        S1 = torch.relu(self.W2(S))
        e = self.W3(S1)

        if self.gated:
            gate = torch.sigmoid(self.U(X))
            e = e * gate

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
        num_heads: int = 4,
        d_attn: Optional[Union[int]] = None,
        temperature: float = 1.0,
        gated: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_heads >= 1
        self.d_model = d_model
        self.num_heads = num_heads

        if d_attn is None:
            assert d_model % num_heads == 0, (
                f"d_model={d_model} must be divisible by num_heads={num_heads}, "
                "or set d_attn explicitly"
            )
            d_attn = d_model // num_heads
        assert num_heads * d_attn == d_model, (
            f"Expected num_heads * d_attn == d_model, got {num_heads} * {d_attn} != {d_model}"
        )

        self.heads = nn.ModuleList(
            [
                VectorAttentionHead(d_model, d_attn, temperature=temperature, gated=gated)
                for _ in range(num_heads)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for head in self.heads:
            head.reset_parameters()

    def forward(self, X: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        vs = []
        for head in self.heads:
            v_i, _ = head(X, attention_mask)
            vs.append(v_i)
        pooled = torch.cat(vs, dim=-1)
        return pooled
