import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Iterable, List, Optional, Dict

# ---------- 温度可调的连续门控（Concrete / Hard-Sigmoid） ----------
class Gate(nn.Module):
    """
    生成 [0,1] 的连续掩码；参数是未约束的 logits。
    temperature 控制平滑度；mode='sigmoid' 更稳定，'concrete' 接近 0/1。
    """
    def __init__(self, size: int, temperature: float = 1.0, mode: str = "sigmoid"):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(size))   # 初始化偏向“保留”（≈0.5）
        self.temperature = temperature
        self.mode = mode

    def forward(self):
        if self.mode == "sigmoid":
            return torch.sigmoid(self.logits)           # [H]
        elif self.mode == "concrete":
            # Concrete gate：更接近二值，同时可微
            return torch.clamp(torch.sigmoid(self.logits / self.temperature), 0.0, 1.0)
        else:
            raise ValueError("mode must be 'sigmoid' or 'concrete'")
        

# ---------- 产生按 hidden 维的掩码（可选 token 聚合） ----------
class MaskNet(nn.Module):
    """
    输入：layer hidden states H ∈ [B, T, H]
    输出：per-hidden-dim gate m ∈ [H] 或 per-token-per-hidden gate m ∈ [T, H]
    """
    def __init__(self, hidden_dim: int, per_token: bool = False, temperature: float = 1.0, mode: str = "sigmoid"):
        super().__init__()
        self.per_token = per_token
        self.temperature = temperature
        self.mode = mode
        if per_token:
            # 若 T 可变，用一个小 MLP 先把 token-wise 压成 T×H 的 gate，再 sigmoid
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
        else:
            self.gate = Gate(hidden_dim, temperature=temperature, mode=mode)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: [B, T, H]
        if self.per_token:
            # 逐 token 计算 gate：先把每个 token 的隐向量过 MLP 得到 logits，再 sigmoid
            logits = self.proj(H)                       # [B, T, H]
            m = torch.sigmoid(logits)                   # [B, T, H]
        else:
            # 对 token 做平均池化，再产生 per-hidden gate
            m_hidden = self.gate()                      # [H]
            m = m_hidden.view(1, 1, -1)                 # broadcast 到 [B, T, H]
        return m


# ---------- Hook：在指定层输出处注入掩码 ----------
class LayerMaskHook:
    """
    针对返回 Tensor 或 tuple(list)[0] 为 [B,T,H] 的层输出做掩码。
    """
    def __init__(self, module: nn.Module, mask_net: MaskNet, lmbda_l1: float = 1e-3):
        self.module = module
        self.mask_net = mask_net
        self.lmbda_l1 = lmbda_l1
        self.handle = None
        self.latest_mask: Optional[torch.Tensor] = None  # 记录最近一次前向的 mask（用于可视化/正则）

    def _hook(self, _module, _inp, out):
        # 标准化拿到张量 H ∈ [B,T,H]
        if isinstance(out, torch.Tensor):
            H = out
            pack = False
        elif isinstance(out, (tuple, list)) and isinstance(out[0], torch.Tensor):
            H = out[0]
            pack = True
        else:
            raise TypeError("Unexpected layer output; need Tensor or tuple with Tensor at [0].")

        # 计算掩码并施加
        m = self.mask_net(H)                # [B,T,H] 或 broadcastable
        self.latest_mask = m
        H_gated = H * m                     # 应用门控

        if pack:
            out = (H_gated,) + tuple(out[1:])
        else:
            out = H_gated
        return out

    def register(self):
        assert self.handle is None, "Hook already registered"
        self.handle = self.module.register_forward_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def l1_regularizer(self) -> torch.Tensor:
        if self.latest_mask is None:
            return torch.tensor(0.0, device=next(self.mask_net.parameters()).device)
        return self.lmbda_l1 * self.latest_mask.mean()   # 越小越稀疏（也可用 .sum()/numel）


# ---------- 训练：最小化 (带掩码输出-原输出)^2 + L1 ----------
class GraphMaskLikeExplainer:
    """
    冻结原模型，仅训练 mask_net，使 gated 输出接近 baseline 输出且 mask 稀疏。
    适用于分类或回归：loss 可用 MSE 于 logit/回归值上；分类亦可在 softmax 前对齐。
    """
    def __init__(self, model: nn.Module, layer_for_mask: nn.Module,
                 hidden_dim: int, per_token: bool = False,
                 lmbda_l1: float = 1e-3, temperature: float = 1.0, mode: str = "sigmoid"):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.mask_net = MaskNet(hidden_dim, per_token=per_token, temperature=temperature, mode=mode)
        self.hook = LayerMaskHook(layer_for_mask, self.mask_net, lmbda_l1=lmbda_l1)

    @contextmanager
    def _hooked(self):
        self.hook.register()
        try:
            yield
        finally:
            self.hook.remove()

    @torch.no_grad()
    def _baseline_outputs(self, dataloader, forward_fn):
        outs = []
        for batch in dataloader:
            out = forward_fn(self.model, batch)         # Tensor [B,C] or [B,1]
            outs.append(out.detach().cpu())
        return torch.cat(outs, dim=0)

    def fit(self, dataloader, forward_fn, epochs: int = 5, lr: float = 1e-2, device: str = "cuda"):
        self.model.to(device)
        self.mask_net.to(device)

        # 1) 基线输出（无掩码）
        baseline = self._baseline_outputs(dataloader, forward_fn).to(device)

        # 2) 训练 mask_net（仅其参数可训练）
        opt = torch.optim.Adam(self.mask_net.parameters(), lr=lr)

        with self._hooked():   # 注册 hook，使 forward 时自动施加掩码
            idx0 = 0
            for ep in range(1, epochs + 1):
                idx0 = 0
                for batch in dataloader:
                    opt.zero_grad()
                    # 带掩码的输出
                    out_gated = forward_fn(self.model, batch)     # [B, …]
                    bsz = out_gated.size(0)

                    # 重建损失：对齐到 baseline 对应切片
                    target = baseline[idx0: idx0 + bsz].to(device)
                    idx0 += bsz

                    # 用 MSE（回归/分类 logits 均可）
                    rec_loss = F.mse_loss(out_gated, target)

                    # 稀疏正则：尽量把 mask 推向 0
                    l1 = self.hook.l1_regularizer()

                    loss = rec_loss + l1
                    loss.backward()
                    opt.step()

                # 每个 epoch 打印统计
                with torch.no_grad():
                    mean_m = self.hook.latest_mask.mean().item() if self.hook.latest_mask is not None else float('nan')
                print(f"[Epoch {ep}] rec={rec_loss.item():.6f} | l1={l1.item():.6f} | mask_mean={mean_m:.4f}")

    @torch.no_grad()
    def get_mask(self, dataloader, reduce: str = "mean") -> torch.Tensor:
        """
        汇总全数据的掩码。
        - per_token=False: 返回 [H]（训练时就固定）
        - per_token=True : 返回按 batch 聚合后的 [T,H]（不同 batch 的 T 需一致）
        """
        self.model.eval()
        masks = []
        with self._hooked():
            for batch in dataloader:
                _ = self.model(**batch) if isinstance(batch, dict) else self.model(*batch)
                m = self.hook.latest_mask.detach()      # [B,T,H] or broadcastable
                masks.append(m)
        M = torch.cat(masks, dim=0)                      # [N,T,H] 或 [N,1, H]
        if reduce == "mean":
            return M.mean(dim=0)                        # -> [T,H] 或 [1,H]
        elif reduce == "none":
            return M
        else:
            raise ValueError("reduce must be 'mean' or 'none'")
