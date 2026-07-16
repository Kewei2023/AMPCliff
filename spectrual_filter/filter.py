# maintained by kewei li
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


def filter_dc(hidden_states, dim, mode, opposite):

    B, T, d = hidden_states.shape
    # assert mask_1d.shape[0] == d
    # mask = mask_1d.to(H.device)
    if dim.lower() == 'seq_len':
        if mode.lower() == 'dct':
            C = dct_ortho(hidden_states, dim=1)  
            # ipdb.set_trace()
            if opposite:
                
                C[:,1:,:]= 0  
            else:
                C[:,0,:]= 0  
            # ipdb.set_trace()
            hidden_states_f = idct_ortho(C, dim=1) 
        if mode.lower() == 'fft':
            C = torch.fft.rfft(hidden_states, dim=1, n=T, norm="backward")
            if opposite:
                C[:,1:,:]= 0  
            else:
                C[:,0,:]= 0  
            hidden_states_f = torch.fft.irfft(C, dim=1) 
    
    if dim.lower() == 'hidden_dim':
        if mode.lower() == 'dct':
            C = dct_ortho(hidden_states, dim=-1) 
            
            if opposite:  
                C[:,:,0]= 0 
            else:
                C[:,:,1:]= 0                 
            hidden_states_f = idct_ortho(C, dim=-1)
        if mode.lower() == 'fft':
            C = torch.fft.rfft(hidden_states, dim=-1, n=d, norm="backward")
            if opposite:  
                C[:,:,0]= 0 
            else:
                C[:,:,1:]= 0   
            hidden_states_f = torch.fft.irfft(C, dim=-1) 

    return hidden_states_f

def scale_dc(hidden_states, dim, mode, scale):

    B, T, d = hidden_states.shape
    # assert mask_1d.shape[0] == d
    # mask = mask_1d.to(H.device)
    if dim.lower() == 'seq_len':
        if mode.lower() == 'dct':
            C = dct_ortho(hidden_states, dim=1)  
            C[:,0,:]*= scale
            hidden_states_f = idct_ortho(C, dim=1) 
        if mode.lower() == 'fft':
            C = torch.fft.rfft(hidden_states, dim=1, n=T, norm="backward")
            C[:,0,:]*= scale
            hidden_states_f = torch.fft.irfft(C, dim=1) 
    
    if dim.lower() == 'hidden_dim':
        if mode.lower() == 'dct':
            C = dct_ortho(hidden_states, dim=-1) 
            
            C[:,0,:]*= scale              
            hidden_states_f = idct_ortho(C, dim=-1)
        if mode.lower() == 'fft':
            C = torch.fft.rfft(hidden_states, dim=-1, n=d, norm="backward")
            C[:,0,:]*= scale 
            hidden_states_f = torch.fft.irfft(C, dim=-1) 

    return hidden_states_f


def concat_dc(hidden_states, dim, mode):

    B, T, d = hidden_states.shape
    # assert mask_1d.shape[0] == d
    # mask = mask_1d.to(H.device)
    if dim.lower() == 'seq_len':
        if mode.lower() == 'dct':
            C = dct_ortho(hidden_states, dim=1) 
            # C_org = C.clone() 
            VC = C.clone() 
            DC = C.clone() 
            DC[:,1:,:]=0
            VC[:,0,:]=0
            # ipdb.set_trace()
            hidden_states_dc = idct_ortho(DC, dim=1) 
            hidden_states_vc = idct_ortho(VC, dim=1) 
        if mode.lower() == 'fft':
            C = torch.fft.rfft(hidden_states, dim=1, n=T, norm="backward")
            VC = C.clone() 
            DC = C.clone()
            DC[:,1:,:]=0
            VC[:,0,:]=0
            hidden_states_dc = torch.fft.irfft(DC, dim=1) 
            hidden_states_vc = torch.fft.irfft(VC, dim=1) 
    
    if dim.lower() == 'hidden_dim':
        if mode.lower() == 'dct':
            C = dct_ortho(hidden_states, dim=-1) 
            VC = C.clone() 
            DC = C.clone()
            DC[:,1:,:]=0
            VC[:,0,:]=0
            hidden_states_dc = idct_ortho(DC, dim=-1) 
            hidden_states_vc = idct_ortho(VC, dim=-1)

        if mode.lower() == 'fft':
            C = torch.fft.rfft(hidden_states, dim=-1, n=d, norm="backward")
            VC = C.clone() 
            DC = C.clone()
            DC[:,1:,:]=0
            VC[:,0,:]=0
            hidden_states_dc = torch.fft.irfft(DC, dim=-1) 
            hidden_states_vc = torch.fft.irfft(VC, dim=-1)

    hidden_states_f = torch.cat([hidden_states_vc,hidden_states_dc], dim=-1)
    return hidden_states_f


def distill_vc(hidden_states, dim, mode, scale):

    B, T, d = hidden_states.shape
    # assert mask_1d.shape[0] == d
    # mask = mask_1d.to(H.device)
    if dim.lower() == 'seq_len':
        if mode.lower() == 'dct':
            C = dct_ortho(hidden_states, dim=1) 
            # C_org = C.clone() 
            VC = C.clone() 
            DC = C.clone() 
            DC[:,1:,:]=0
            VC[:,0,:]=0
            VC *= scale
            # ipdb.set_trace()
            hidden_states_dc = idct_ortho(DC, dim=1) 
            hidden_states_vc = idct_ortho(VC, dim=1) 
        if mode.lower() == 'fft':
            C = torch.fft.rfft(hidden_states, dim=1, n=T, norm="backward")
            VC = C.clone() 
            DC = C.clone()
            DC[:,1:,:]=0
            VC[:,0,:]=0
            VC *= scale
            hidden_states_dc = torch.fft.irfft(DC, dim=1) 
            hidden_states_vc = torch.fft.irfft(VC, dim=1) 
    
    if dim.lower() == 'hidden_dim':
        if mode.lower() == 'dct':
            C = dct_ortho(hidden_states, dim=-1) 
            VC = C.clone() 
            DC = C.clone()
            DC[:,1:,:]=0
            VC[:,0,:]=0
            VC *= scale
            hidden_states_dc = idct_ortho(DC, dim=-1) 
            hidden_states_vc = idct_ortho(VC, dim=-1)

        if mode.lower() == 'fft':
            C = torch.fft.rfft(hidden_states, dim=-1, n=d, norm="backward")
            VC = C.clone() 
            DC = C.clone()
            DC[:,1:,:]=0
            VC[:,0,:]=0
            VC *= scale
            hidden_states_dc = torch.fft.irfft(DC, dim=-1) 
            hidden_states_vc = torch.fft.irfft(VC, dim=-1)

    hidden_states_f = torch.cat([hidden_states_vc,hidden_states_dc], dim=-1)
    return hidden_states_f

def seperate_dc(hidden_states, dim, mode):

    B, T, d = hidden_states.shape
    # assert mask_1d.shape[0] == d
    # mask = mask_1d.to(H.device)
    if dim.lower() == 'seq_len':
        if mode.lower() == 'dct':
            C = dct_ortho(hidden_states, dim=1) 
            # C_org = C.clone() 
            VC = C.clone() 
            DC = C.clone() 
            DC[:,1:,:]=0
            VC[:,0,:]=0
            
            # ipdb.set_trace()
            hidden_states_dc = idct_ortho(DC, dim=1) 
            hidden_states_vc = idct_ortho(VC, dim=1) 
        if mode.lower() == 'fft':
            C = torch.fft.rfft(hidden_states, dim=1, n=T, norm="backward")
            VC = C.clone() 
            DC = C.clone()
            DC[:,1:,:]=0
            VC[:,0,:]=0
            # VC *= scale
            hidden_states_dc = torch.fft.irfft(DC, dim=1) 
            hidden_states_vc = torch.fft.irfft(VC, dim=1) 
    
    if dim.lower() == 'hidden_dim':
        if mode.lower() == 'dct':
            C = dct_ortho(hidden_states, dim=-1) 
            VC = C.clone() 
            DC = C.clone()
            DC[:,1:,:]=0
            VC[:,0,:]=0
            # VC *= scale
            hidden_states_dc = idct_ortho(DC, dim=-1) 
            hidden_states_vc = idct_ortho(VC, dim=-1)

        if mode.lower() == 'fft':
            C = torch.fft.rfft(hidden_states, dim=-1, n=d, norm="backward")
            VC = C.clone() 
            DC = C.clone()
            DC[:,1:,:]=0
            VC[:,0,:]=0
            # VC *= scale
            hidden_states_dc = torch.fft.irfft(DC, dim=-1) 
            hidden_states_vc = torch.fft.irfft(VC, dim=-1)

    # hidden_states_f = torch.cat([hidden_states_vc,hidden_states_dc], dim=-1)
    return hidden_states_dc,hidden_states_vc

# -------------------- 按论文 Spectral Band Allocation 生成频带切分 --------------------
def allocate_prism_bands(
    n: int,
    k: int = 5,
    base: int = 4,
    mode: str = "uniform",          # 'geometric' | 'uniform'
    device: torch.device = None,
    mask_dtype: torch.dtype = torch.float32,
):
    """
    将 n 个频率索引 [0..n-1] 划成 k 个相邻频带。
    返回:
      masks  : list[Tensor[n]]  每个频带的 0/1 掩膜（按顺序）
      sizes  : LongTensor[k]    每带长度
      starts : LongTensor[k]    每带起点（含）
      ends   : LongTensor[k]    每带终点（不含）
    mode:
      - 'uniform'  : 等宽划分（尽量均分，余数前 r 段各 +1）
      - 'geometric': 几何比例（低频窄、高频宽，s_i = base**i）
    """
    assert n >= k >= 1
    if device is None:
        device = torch.device("cpu")
    mode = mode.lower()

    if mode == "uniform":
        # 基本宽度 & 余数
        q, r = divmod(n, k)                   # 前 r 段长 q+1，后面 k-r 段长 q
        sizes = torch.tensor([q + 1]*r + [q]*(k - r), dtype=torch.long, device=device)

    elif mode == "geometric":
        # 先给每段至少 1，然后按 s_i = base**i 分配剩余
        sizes = torch.ones(k, dtype=torch.long, device=device)
        remaining = n - k
        i = torch.arange(k, dtype=torch.float32, device=device)
        s = (base ** i)                                   # 比例分配权重
        frac  = remaining * (s / s.sum())                 # 理想的分数分配
        floor = torch.floor(frac).to(torch.long)          # 先取整
        sizes += floor
        left = int(remaining - int(floor.sum()))          # 还剩多少需要补 1
        if left > 0:
            residual = (frac - floor.float()).tolist()
            order = sorted(range(k), key=lambda t: residual[t], reverse=True)
            for j in range(left):
                sizes[order[j]] += 1
    else:
        raise ValueError(f"mode must be 'uniform' or 'geometric', got '{mode}'")

    # 边界与掩膜
    ends   = torch.cumsum(sizes, dim=0)
    starts = torch.cat([torch.tensor([0], dtype=torch.long, device=device), ends[:-1]])
    masks = []
    for st, ed in zip(starts.tolist(), ends.tolist()):
        m = torch.zeros(n, dtype=mask_dtype, device=device)
        m[st:ed] = 1.0
        masks.append(m)

    return masks, sizes, starts, ends

def make_prism_band_mask_d(d: int, band_index: int, k: int = 5, base: int = 4,mode:str = 'uniform',device: torch.device = None):
    """取第 band_index 个频带(0..k-1; 0=LOW, k-1=HIGH) 的 0/1 掩膜（长度 d）。"""
    masks, _, _, _ = allocate_prism_bands(d, k=k, base=base,mode=mode,device=device)
    assert 0 <= band_index < k
    return masks[band_index]

# -------------------- 在 hidden_dim 上做频带滤波（band-pass or notch） --------------------
def spectral_filter_hidden_dim(H: torch.Tensor,
                               mask_1d: torch.Tensor,
                               mode: str = 'pass',
                               preserve_norm: bool = False,
                            #    rotary_knock: list = [],
                            #    collect_energy: bool = False,     # 打开统计
                            #    energy_store: list = None
                               ) -> torch.Tensor:
    """
    H: [B, T, d]；mask_1d: [d] 的 0/1 向量（来自 make_prism_band_mask_d）
    mode: 'pass' 仅保留该频带；'notch' 去掉该频带
    preserve_norm: 对 notch 模式按 token 的 L2 比例缩放，降低尺度漂移
    """
    B, T, d = H.shape
    assert mask_1d.shape[0] == d
    mask = mask_1d.to(H.device)
    C = dct_ortho(H, dim=-1)                         # 仅沿 hidden_dim
    C_f = C * mask if mode == 'pass' else C * (1.0 - mask)
    H_f = idct_ortho(C_f, dim=-1)

    if preserve_norm and mode == 'notch':
        eps = 1e-8
        s = (H.norm(p=2, dim=-1, keepdim=True) + eps) / \
            (H_f.norm(p=2, dim=-1, keepdim=True) + eps)
        H_f = H_f * s
    return H_f

# -------------------- Hook：把频带滤波注入到 HuggingFace EsmModel 的指定层 --------------------
def _spectral_on_hidden_dim(H: torch.Tensor,
                            band_index: int, k_bands: int, base: int,
                            mode: str = 'notch', band_mode: str = 'uniform', 
                            preserve_norm: bool = True,# rotary_knock: list = [],
                            _mask_cache: dict = None) -> torch.Tensor:
    assert H.dim() == 3, f"expected hidden [B,T,H], got {H.shape}"
    d = H.size(-1)
    key = (d, k_bands, base, band_index, H.device)
    if _mask_cache is not None and key in _mask_cache:
        mask = _mask_cache[key]
    else:
        mask = make_prism_band_mask_d(d, band_index, k=k_bands, base=base,mode=band_mode, device=H.device)
        if _mask_cache is not None:
            _mask_cache[key] = mask
    H_new = spectral_filter_hidden_dim(H, mask, mode=mode, preserve_norm=preserve_norm)
    return H_new


class HiddenDimPrismHookHF:
    """
    专为 HF 的 EsmLayer.forward 返回 (layer_output, ...) 设计：
    仅对 outputs[0] (shape=[B,T,H]) 做 hidden_dim 频带处理，其它元素原样透传。
    """
    def __init__(self, hf_esm,  # 传入 model.backbone.encoder
                 target_layers,       # 如 [30] 或 range(num_layers)
                 band_index: int,
                 k_bands: int = 5,
                 base: int = 4,
                 band_mode: str = 'uniform',
                 mode: str = 'notch',        # 'pass' or 'notch'
                 rotary_knock: list = [],
                 preserve_norm: bool = True):
        self.encoder = hf_esm
        self.target_layers = set(int(i) for i in target_layers)
        self.band_index = int(band_index)
        self.k_bands = int(k_bands)
        self.base = int(base)
        self.mode = str(mode)
        self.band_mode = str(band_mode)
        self.preserve_norm = bool(preserve_norm)
        self.handles = []
        self.rotary_knock = []
        self._mask_cache = {}

    def _hook_fn(self, layer_id: int):
        def fn(module, inputs, outputs):
            # EsmLayer.forward 返回的是 tuple: (layer_output, ...) 
            if not isinstance(outputs, (tuple, list)):
                raise TypeError(f"[hook@layer {layer_id}] unexpected output type: {type(outputs)}")
            if len(outputs) < 1 or not isinstance(outputs[0], torch.Tensor):
                raise TypeError(f"[hook@layer {layer_id}] outputs[0] is not a Tensor")

            H = outputs[0]                           # [B,T,H]
            H_new = _spectral_on_hidden_dim(
                H,
                band_index=self.band_index,
                k_bands=self.k_bands,
                base=self.base,
                band_mode=self.band_mode,
                mode=self.mode,
                preserve_norm=self.preserve_norm,
                # rotary_knock=self.rotary_knock,
                _mask_cache=self._mask_cache
            )
            assert H_new.shape == H.shape, f"[hook@layer {layer_id}] shape changed: {H_new.shape} vs {H.shape}"
            # 用同类型（tuple/list）返回，替换第 0 个元素为 H_new
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
