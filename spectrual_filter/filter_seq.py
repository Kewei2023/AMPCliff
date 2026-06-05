import torch
import torch.nn as nn
from transformers import EsmModel
import ipdb
# -------------------- DCT/IDCT（优先用 torch-dct; 失败时退化到矩阵基实现） --------------------
from torch_dct import dct as _dct_1d, idct as _idct_1d

def _move_apply(fn, x: torch.Tensor, dim: int):
    """把 dim 维移到最后一维 -> 调用 fn(x_lastdim) -> 移回 dim。"""
    if dim < 0:
        dim = x.ndim + dim
    if dim == x.ndim - 1:
        return fn(x)                      # 已在最后一维
    x_perm = x.movedim(dim, -1)
    y = fn(x_perm)
    return y.movedim(-1, dim)

def dct_ortho(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 旧版 torch_dct 没有 dim 参数，因此总是用 move-dim 的方式
    try:
        return _move_apply(lambda t: _dct_1d(t, norm='ortho'), x, dim)
    except TypeError:
        # 极旧版本若也不支持 norm，则去掉 norm（会有全局尺度差异）
        return _move_apply(lambda t: _dct_1d(t), x, dim)

def idct_ortho(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    try:
        return _move_apply(lambda t: _idct_1d(t, norm='ortho'), x, dim)
    except TypeError:
        return _move_apply(lambda t: _idct_1d(t), x, dim)
# -------------------- Prism 频带分配（论文规则） --------------------
def allocate_prism_bands(n: int, k: int = 5, base: int = 4):
    assert n >= k >= 1
    sizes = torch.ones(k, dtype=torch.long)
    remaining = n - k
    i = torch.arange(k, dtype=torch.float32)
    s = (base ** i)
    frac = remaining * (s / s.sum())
    floor = torch.floor(frac).to(torch.long)
    sizes += floor
    left = int(remaining - int(floor.sum()))
    residual = (frac - floor.float()).tolist()
    order = sorted(range(k), key=lambda t: residual[t], reverse=True)
    for j in range(left):
        sizes[order[j]] += 1
    ends = torch.cumsum(sizes, dim=0)
    starts = torch.cat([torch.tensor([0], dtype=torch.long), ends[:-1]])
    masks = []
    for st, ed in zip(starts.tolist(), ends.tolist()):
        m = torch.zeros(n); m[st:ed] = 1.0
        masks.append(m)
    return masks, sizes, starts, ends

def make_prism_band_mask_len(n: int, band_index: int, k: int = 5, base: int = 4):
    masks, _, _, _ = allocate_prism_bands(n, k=k, base=base)
    assert 0 <= band_index < k
    return masks[band_index]   # [n]

# -------------------- 序列维（token 维）频带滤波 --------------------
def spectral_filter_seq_dim(H: torch.Tensor,
                            lengths: torch.Tensor,     # [B] 每个样本有效长度 L_b
                            band_index: int,
                            k_bands: int = 5,
                            base: int = 4,
                            mode: str = 'notch',       # 'pass' or 'notch'
                            preserve_norm: bool = True) -> torch.Tensor:
    """
    对 H[b, :L_b, :] 在“序列维 dim=1”做 DCT -> 掩膜 -> IDCT；padding 段不改。
    H: [B, T, d]
    lengths: [B]（int），每个样本有效 token 数
    """
    assert H.dim() == 3
    B, T, d = H.shape
    H_new = H.clone()

    for b in range(B):
        L = int(lengths[b].item())
        if L <= 1:
            continue
        x = H[b, :L, :]                       # [L, d]
        C = dct_ortho(x, dim=0)               # 沿序列维（第 0 维）做 DCT ⇒ [L, d]
        mask = make_prism_band_mask_len(L, band_index, k=k_bands, base=base).to(H.device)  # [L]
        if mode == 'pass':
            C_f = C * mask.view(L, 1)
        elif mode == 'notch':
            C_f = C * (1.0 - mask.view(L, 1))
        else:
            raise ValueError("mode must be 'pass' or 'notch'")
        x_f = idct_ortho(C_f, dim=0)          # [L, d]

        if preserve_norm and mode == 'notch':
            eps = 1e-8
            s = (x.norm(p=2, dim=-1, keepdim=True) + eps) / (x_f.norm(p=2, dim=-1, keepdim=True) + eps)
            x_f = x_f * s

        H_new[b, :L, :] = x_f                 # padding 段不动

    return H_new

# -------------------- Hook（适配 EsmLayer.forward 返回 (layer_output, ...)） --------------------
def _infer_lengths_from_attention_mask(attn_mask, T, device):
    """
    attn_mask 通常是扩展后的 [B,1,1,T]，有效 token 位置为 0，padding 为负大数。
    若拿不到，回退为全长 T。
    """
    if isinstance(attn_mask, torch.Tensor) and attn_mask.dim() == 4 and attn_mask.size(-1) == T:
        # 有效为 0，padding 为负值
        valid = (attn_mask.squeeze(1).squeeze(1) == 0)  # [B,T] bool
        lengths = valid.sum(dim=1)
        # print("[DEBUG] available lengths:",lengths)
        return lengths.to(device)
    else:
        # 回退：认为全长有效
        B = attn_mask.size(0) if isinstance(attn_mask, torch.Tensor) else None
        if B is None:
            raise RuntimeError("Cannot infer lengths: attention_mask missing in EsmLayer inputs.")
        return torch.full((B,), T, dtype=torch.long, device=device)

class HiddenDimPrismHookHF:
    """
    在 HuggingFace 的 EsmLayer 层输出处，对 outputs[0]（[B,T,d]）沿“序列维”做 DCT 频带处理。
    其它输出（注意力、缓存）保持不变。
    """
    def __init__(self, hf_esm,           # 传入 model.backbone.encoder 或 DDP 下的 model.module.backbone.encoder
                 target_layers,               # 例如 range(num_layers) 或 [30]
                 band_index: int,
                 k_bands: int = 5,
                 base: int = 4,
                 mode: str = 'notch',         # 'pass' or 'notch'
                 preserve_norm: bool = True):
        self.encoder = hf_esm
        self.target_layers = set(int(i) for i in target_layers)
        self.band_index = int(band_index)
        self.k_bands = int(k_bands)
        self.base = int(base)
        self.mode = str(mode)
        self.preserve_norm = bool(preserve_norm)
        self.handles = []

    def _hook_fn(self, layer_id: int):
        def fn(module, inputs, outputs):
            # EsmLayer.forward: inputs = (hidden_states, attention_mask, head_mask, ...)
            # outputs: (layer_output, ...) 其中 layer_output 是 [B,T,d]
            if not isinstance(outputs, (tuple, list)):
                raise TypeError(f"[seq-hook@{layer_id}] unexpected outputs type: {type(outputs)}")
            H = outputs[0]
            assert isinstance(H, torch.Tensor) and H.dim() == 3, \
                f"[seq-hook@{layer_id}] outputs[0] must be [B,T,d], got {type(H)} with shape {getattr(H,'shape',None)}"

            # 取 attention_mask 推断每个样本有效长度
            attn_mask = inputs[1] if len(inputs) >= 2 else None
            lengths = _infer_lengths_from_attention_mask(attn_mask, H.size(1), H.device)  # [B]

            # 在序列维做 Prism 频带滤波
            H_new = spectral_filter_seq_dim(
                H, lengths,
                band_index=self.band_index, k_bands=self.k_bands, base=self.base,
                mode=self.mode, preserve_norm=self.preserve_norm
            )
            assert H_new.shape == H.shape

            if isinstance(outputs, tuple):
                return (H_new,) + outputs[1:]
            else:
                out_list = list(outputs)
                out_list[0] = H_new
                return out_list
        return fn

    def register(self):
        layers = self.encoder.layer
        for lid, layer in enumerate(layers):
            if lid in self.target_layers:
                h = layer.register_forward_hook(self._hook_fn(lid))
                self.handles.append(h)

    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()