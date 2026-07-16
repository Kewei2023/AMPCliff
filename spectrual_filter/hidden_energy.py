# maintained by kewei li
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import ipdb
# ========== DCT（兼容旧版 torch_dct 无 dim 形参） ==========
try:
    from torch_dct import dct as _dct_1d
    def _move_apply(fn, x: torch.Tensor, dim: int):
        if dim < 0: dim = x.ndim + dim
        if dim == x.ndim - 1:
            return fn(x)
        x_perm = x.movedim(dim, -1)
        y = fn(x_perm)
        return y.movedim(-1, dim)
    def dct_ortho(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        try:
            return _move_apply(lambda t: _dct_1d(t, norm='ortho'), x, dim)
        except TypeError:
            return _move_apply(lambda t: _dct_1d(t), x, dim)
except Exception:
    _DCT_CACHE = {}
    def _dct_matrix(N, device):
        k = torch.arange(N, device=device).float()
        n = torch.arange(N, device=device).float()
        M = torch.cos(torch.pi * (n[:, None] + 0.5) * k[None, :] / N)
        M[:, 0] *= (1.0/torch.sqrt(torch.tensor(2.0, device=device)))
        M *= torch.sqrt(2.0 / N)
        return M
    def dct_ortho(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        N = x.size(dim); key=(N, x.device)
        if key not in _DCT_CACHE: _DCT_CACHE[key] = _dct_matrix(N, x.device)
        M = _DCT_CACHE[key]
        xT = x.transpose(dim, -1)
        yT = xT @ M
        return yT.transpose(dim, -1)

# ========== Prism：几何比例自动分段（用于画虚线标注） ==========
def prism_band_edges(n: int, k: int = 5, base: int = 4, mode: str = "uniform"):
    """
    生成频带分界。
    mode:
      - 'geometric': 几何比例（低频窄高频宽）
      - 'uniform'  : 等宽划分
    返回 [(start, end)] * k，为半开区间。
    """
    assert n >= k >= 1
    mode = mode.lower()
    
    if mode == "uniform":
        step = n // k
        edges = [(i * step, (i + 1) * step if i < k - 1 else n) for i in range(k)]
        return edges

    elif mode == "geometric":
        sizes = torch.ones(k, dtype=torch.long)
        remaining = n - k
        i = torch.arange(k, dtype=torch.float32)
        s = (base ** i)
        frac  = remaining * (s / s.sum())
        floor = torch.floor(frac).to(torch.long)
        sizes += floor
        left = int(remaining - int(floor.sum()))
        residual = (frac - floor.float()).tolist()
        order = sorted(range(k), key=lambda t: residual[t], reverse=True)
        for j in range(left):
            sizes[order[j]] += 1
        ends = torch.cumsum(sizes, dim=0)
        starts = torch.cat([torch.tensor([0], dtype=torch.long), ends[:-1]])
        return [(int(starts[u]), int(ends[u])) for u in range(k)]

    else:
        raise ValueError(f"Unknown mode '{mode}', must be 'geometric' or 'uniform'")

# ========== 频率级能量占比：逐层、逐频率 k ==========
@torch.no_grad()
def band_energy_share_per_layer_4d(
    X: torch.Tensor,                      # [N, T, H, L]
    token_valid_mask: torch.Tensor = None,# 可选 [N, T]（有效=1,pad=0）
    batch_size: int = 64,
    device: str = None,
    dim: str = 'hidden_dim',
    exclude_dc: bool = False              # 如需排除 DC(k=0) 再归一化
):
    """
    沿 hidden_dim(H) 做 DCT；在序列维(T) 聚合能量；对每个样本在 H 维归一化（可选去 DC）。
    返回 df_long（列: layer, k, share）与 H。
    """
    assert X.dim() == 4, f"expected [N,T,H,L], got {tuple(X.shape)}"
    N, T, H, L = X.shape
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)

    if token_valid_mask is not None:
        assert token_valid_mask.shape[:2] == (N, T)
        m_all = token_valid_mask.to(device).to(X.dtype).unsqueeze(-1).unsqueeze(-1)  # [N,T,1,1]
    else:
        m_all = None


    if dim == 'hidden_dim':
        dim_to_dct = 2
        dim_to_dct_sum = 1
        Num = H
    if dim == 'seq_len':
        dim_to_dct = 1
        dim_to_dct_sum = 2
        Num = T
    records = []
    for s in range(0, N, batch_size):
        e  = min(N, s + batch_size)
        Xb = X[s:e]                      # [B,T,H,L]
        B  = Xb.size(0)
        mb = m_all[s:e] if m_all is not None else None
        # ipdb.set_trace()
        for l in range(L):
            Xt = Xb[..., l]              # [B,T,H]
            C  = dct_ortho(Xt, dim=dim_to_dct)    # [B,T,H] 沿 hidden_dim
            E  = C**2                    # 能量
            if mb is not None:
                E = E * mb[..., 0, 0]    # [B,T,H]（mb原是 [B,T,1,1]）

            Eh = E.sum(dim=dim_to_dct_sum)            # [B,H/T]，沿 T/H 聚合
            if exclude_dc:
                Eh_dc0 = Eh.clone()
                Eh_dc0[:, 0] = 0.0
                denom = Eh_dc0.sum(dim=1, keepdim=True) + 1e-12
                share = Eh_dc0 / denom
            else:
                denom = Eh.sum(dim=1, keepdim=True) + 1e-12
                share = Eh / denom       # [B,H]
            # ipdb.set_trace()
            df = pd.DataFrame(share.detach().cpu().numpy(), columns=[f"k{j}" for j in range(Num)])
            df = df.melt(var_name="freq", value_name="share")
            df["k"] = df["freq"].str.replace("k", "", regex=False).astype(int)
            df["layer"] = l
            records.append(df[["layer", "k", "share"]])

    df_long = pd.concat(records, ignore_index=True)
    # ipdb.set_trace()
    return df_long, Num

# ========== 画图：频率曲线 + Prism 5段虚线标注 ==========
def plot_band_shares_all_layers(
    df_long: pd.DataFrame,   # 列: layer, k, share
    H: int,
    k_bands: int = 5,
    base: int = 4,
    band_mode: str='uniform',
    title: str = "Hidden-DCT frequency energy share per layer",
    out_png: str = None,
    out_csv_curve: str = None
):
    plt.figure(figsize=(12, 5))
    ax = sns.lineplot(
        data=df_long, x="k", y="share",
        hue="layer", ci="sd", marker=None
    )
    ax.set_xlabel("hidden-frequency index k (DCT along hidden_dim)")
    ax.set_ylabel("energy share")
    ax.set_title(title)

    # Prism 边界（竖直虚线）
    edges = prism_band_edges(H, k=k_bands, base=base,mode=band_mode)
    for (s, e) in edges:
        ax.axvline(s,  color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(e-1, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    # 可选：在顶部写 band 标签
    ymax = ax.get_ylim()[1]
    for bi, (s, e) in enumerate(edges):
        ax.text((s+e-1)/2, ymax*0.95, f"band {bi}\n[{s},{e})",
                ha="center", va="top", fontsize=8, alpha=0.8)

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
    plt.close()

    if out_csv_curve:
        grp  = df_long.groupby(["layer", "k"])["share"]
        mean = grp.mean()
        std  = grp.std(ddof=1)
        cnt  = grp.count().clip(lower=1)
        ci95 = 1.96 * (std / np.sqrt(cnt))
        curve = (pd.DataFrame({"mean": mean, "ci95": ci95, "n": cnt})
                 .reset_index()
                 .sort_values(["layer", "k"]))
        curve.to_csv(out_csv_curve, index=False)


def plot_mse_diff_in_groups(
    plot_df: pd.DataFrame,      # 列: ['layer','band','mse_diff']
    k_bands: int = 20,
    group_size: int = 5,        # 每图 5 条线
    base_title: str = "ESM2 spectral notch on {split} (k={k}, base={base})\nMSE difference: (with-filter − baseline)",
    split: str = "valid",
    k: int = 20,
    base: int = 4,
    out_prefix: str = "./mse_diff_bandgroup_",   # 输出文件前缀
    palette_name: str = "tab10",  # 每图 5 条线，tab10 可读性好
    set_common_ylim: bool = True  # 4 张图使用统一 y 轴范围，便于横向比较
):
    assert {"layer","band","mse_diff"} <= set(plot_df.columns)
    # 只保留存在的 band（有时某些 band 为空）
    existing_bands = sorted(plot_df["band"].unique().astype(int).tolist())
    # 分组
    groups = [existing_bands[i:i+group_size] for i in range(0, len(existing_bands), group_size)]

    # 统一 y 轴
    if set_common_ylim:
        y_min = np.floor(plot_df["mse_diff"].min()*100)/100.0
        y_max = np.ceil (plot_df["mse_diff"].max()*100)/100.0
        if y_min == y_max:
            y_min -= 0.01; y_max += 0.01
        ylim = (y_min, y_max)
    else:
        ylim = None

    for gi, bands in enumerate(groups):
        df_g = plot_df[plot_df["band"].isin(bands)].copy()
        if df_g.empty:
            continue

        plt.figure(figsize=(9, 4.5))
        # 给当前 5 条线准备可区分颜色
        palette = sns.color_palette(palette_name, n_colors=len(bands))

        ax = sns.lineplot(
            data=df_g.sort_values(["band","layer"]),
            x="layer", y="mse_diff",
            hue="band", palette=palette,
            linewidth=1.8, marker="o", alpha=0.95, legend="brief"
        )

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("MSE difference")
        title = base_title.format(split=split, k=k, base=base)
        ax.set_title(f"{title}\nBands: {bands}")
        if ylim is not None:
            ax.set_ylim(*ylim)

        # 图例放外侧
        ax.legend(title="band index", bbox_to_anchor=(1.02, 1),
                  loc="upper left", borderaxespad=0.)

        plt.tight_layout()
        out_png = f"{out_prefix}{split}_group{gi+1}_of_{math.ceil(k_bands/group_size)}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[saved] {out_png}") 