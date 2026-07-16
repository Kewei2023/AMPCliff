# maintained by kewei li
import torch
from typing import Optional, Literal, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import ipdb
from AMPCliff.spectrual_filter.filter import allocate_prism_bands


@torch.no_grad()
def fft_layer_freq_distribution(
    x: torch.Tensor,                         # [B, D, L]
    use_rfft: bool = True,                   # True -> rfft, F = floor(D/2)+1
    norm: str = "backward",
    measure: Literal["real","mag","mag2"] = "mag",  # 统计用幅度|X|或功率|X|^2
    topk: int = 1,                           # 统计每层Top-K主导频率的出现次数
    sample_weights: Optional[torch.Tensor] = None,  # [B] 可选样本权重
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    返回字典:
      'F':                 频点数 (int)
      'k':                 [F] 频率索引 (0..F-1)
      'omega':             [F] 数字角频率 2πk/D
      'mean_spectrum':     [L, F] 每层跨batch平均的谱强度 (按 measure)
      'prob_per_layer':    [L, F] 每层对频点的归一化分布 (和=1)
      'dominant_counts':   [L, F] 每层主导频率(Top-1)出现次数直方图
      'topk_counts':       [L, F] 每层Top-K出现次数直方图 (K=topk)
    """
    assert x.ndim == 3, "x must be [B, D, L]"
    B, D, L = x.shape
    orig_dtype = x.dtype

    # --- 数值安全：低精度升到 fp32 ---
    x32 = x.to(torch.float32) if (orig_dtype in (torch.float16, torch.bfloat16) or not x.is_floating_point()) else x

    # --- FFT along hidden_dim (dim=1) ---
    if use_rfft:
        Y = torch.fft.rfft(x32, dim=1, norm=norm)      # [B, F, L] complex
        F = Y.shape[1]
    else:
        Y = torch.fft.fft(x32,  dim=1, norm=norm)      # [B, D, L] complex
        F = D

    # 重排到 [B, L, F] 便于逐层统计
    Y_blf = Y.permute(0, 2, 1).contiguous()            # [B, L, F]
    if measure == "mag":
        S_blf = torch.sqrt(Y_blf.real**2 + Y_blf.imag**2 + eps)  # [B,L,F]
    elif measure == "mag2":
        S_blf = (Y_blf.real**2 + Y_blf.imag**2)                  # [B,L,F]
    elif measure == "real":
        S_blf = Y_blf.real
    else:
        raise ValueError("measure must be 'real', 'mag' or 'mag2'")

    # --- 样本权重 ---
    if sample_weights is not None:
        sample_weights = sample_weights.reshape(-1, 1, 1).to(S_blf.dtype).to(S_blf.device)  # [B,1,1]
        S_weighted = S_blf * sample_weights
        denom = sample_weights.sum(dim=0).clamp_min(eps)  # [1,L,F] 广播成 [1,1,1]→按B求和
    else:
        S_weighted = S_blf
        denom = torch.tensor(B, dtype=S_blf.dtype, device=S_blf.device)

    # --- 每层平均谱与概率分布 ---
    mean_spectrum_lf = S_weighted.sum(dim=0) / denom       # [L,F]
    prob_per_layer = mean_spectrum_lf / (mean_spectrum_lf.sum(dim=1, keepdim=True) + eps)  # [L,F]

    # --- 主导频率直方图（Top-1 或 Top-K）---
    # 取Top-K索引: [B,L,K]
    K = max(1, int(topk))
    K = min(K, F)
    topk_idx = S_blf.topk(K, dim=-1, largest=True).indices  # [B,L,K]

    # 统计出现次数: 对每层分别 bincount
    dominant_counts = torch.zeros(L, F, device=x.device, dtype=torch.long)
    topk_counts     = torch.zeros(L, F, device=x.device, dtype=torch.long)

    # 权重计数（可选）：若样本有权重，改用加权直方图（float），这里先给无权重整型版本
    for ell in range(L):
        # Top-1
        dom1 = topk_idx[:, ell, 0]                         # [B]
        dominant_counts[ell] = torch.bincount(dom1, minlength=F)
        # Top-K（重复频点计数多次）
        tk = topk_idx[:, ell, :].reshape(-1)               # [B*K]
        topk_counts[ell] = torch.bincount(tk, minlength=F)

    # --- 频率坐标 ---
    k = torch.arange(F, device=x.device)
    omega = 2.0 * torch.pi * k.to(torch.float32) / D       # 数字角频率

    return {
        "F": torch.tensor(F),
        "k": k,                                # 0..F-1
        "omega": omega,                         # 2πk/D
        "mean_spectrum": mean_spectrum_lf,     # [L,F]
        "prob_per_layer": prob_per_layer,      # [L,F]
        "dominant_counts": dominant_counts,    # [L,F]
        "topk_counts": topk_counts,            # [L,F]
    }




@torch.no_grad()
def compute_layer_fft_spectra(
    x: torch.Tensor,                         # [B, D, L]
    use_rfft: bool = True,                   # True -> rfft, F = floor(D/2)+1
    norm: str = "backward",
    measure: Literal["mag","mag2","real","imag"] = "mag",  # what to plot
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    assert x.ndim == 3, "x must be [B, D, L]"
    B, D, L = x.shape
    orig_dtype = x.dtype
    x32 = x.to(torch.float32) if (orig_dtype in (torch.float16, torch.bfloat16) or not x.is_floating_point()) else x

    if use_rfft:
        Y = torch.fft.rfft(x32, dim=1, norm=norm)  # [B, F, L] complex
        F = Y.shape[1]
    else:
        Y = torch.fft.fft(x32, dim=1, norm=norm)   # [B, D, L]
        F = D

    Y_blf = Y.permute(0, 2, 1).contiguous()        # [B, L, F]
    if measure == "mag":
        S_blf = torch.sqrt(Y_blf.real**2 + Y_blf.imag**2 + eps)
    elif measure == "mag2":
        S_blf = (Y_blf.real**2 + Y_blf.imag**2)
    elif measure == "real":
        S_blf = Y_blf.real
    elif measure == "imag":
        S_blf = Y_blf.imag
    else:
        raise ValueError("measure must be 'mag','mag2','real','imag'")

    k = torch.arange(F, device=x.device)
    return {"S_blf": S_blf, "k": k, "F": torch.tensor(F), "D": torch.tensor(D), "L": torch.tensor(L)}

def compute_seq_fft_spectra(
    x: torch.Tensor,                         # [B, D, L]
    use_rfft: bool = True,                   # True -> rfft, F = floor(D/2)+1
    norm: str = "backward",
    measure: Literal["mag","mag2","real","imag"] = "mag",  # what to plot
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    assert x.ndim == 3, "x must be [B, D, L]"
    B, T, D = x.shape
    orig_dtype = x.dtype
    x32 = x.to(torch.float32) if (orig_dtype in (torch.float16, torch.bfloat16) or not x.is_floating_point()) else x

    if use_rfft:
        Y = torch.fft.rfft(x32, dim=1, norm=norm)  # [B, F, D] complex
        F = Y.shape[1]
    else:
        Y = torch.fft.fft(x32, dim=1, norm=norm)   # [B, T, D]
        F = D

    Y_blf = Y.permute(0, 2, 1).contiguous()        # [B, D, T ]
    if measure == "mag":
        S_blf = torch.sqrt(Y_blf.real**2 + Y_blf.imag**2 + eps)
    elif measure == "mag2":
        S_blf = (Y_blf.real**2 + Y_blf.imag**2)
    elif measure == "real":
        S_blf = Y_blf.real
    elif measure == "imag":
        S_blf = Y_blf.imag
    else:
        raise ValueError("measure must be 'mag','mag2','real','imag'")

    k = torch.arange(F, device=x.device)
    return {"S_blf": S_blf, "k": k, "F": torch.tensor(F), "D": torch.tensor(D), "L": torch.tensor(L)}

def plot_layer_curves(
    S_blf: torch.Tensor,          # [B,L,F] real-valued spectra to plot
    k: torch.Tensor,              # [F] frequency index
    normalize: Optional[Literal["none","prob","zscore","minmax"]] = "none",
    out_prefix: str = "/mnt/data/layer",
) -> List[str]:
    """
    normalize:
      - 'none'  : raw values
      - 'prob'  : divide by sum over k per (B, L)
      - 'zscore': per (B, L) zero-mean, unit-std
      - 'minmax': per (B, L) (x - min)/(max - min + 1e-8)
    """
    B, L, F = S_blf.shape
    k_np = k.cpu().numpy()
    paths = []

    S = S_blf.clone()
    if normalize == "prob":
        S = S / (S.sum(dim=-1, keepdim=True) + 1e-8)
    elif normalize == "zscore":
        mu = S.mean(dim=-1, keepdim=True)
        sd = S.std(dim=-1, keepdim=True).clamp_min(1e-8)
        S = (S - mu) / sd
    elif normalize == "minmax":
        mn = S.amin(dim=-1, keepdim=True)
        mx = S.amax(dim=-1, keepdim=True)
        S = (S - mn) / (mx - mn + 1e-8)

    S_np = S.cpu().numpy()  # [B,L,F]

    for ell in range(L):
        plt.figure()
        for b in range(B):
            plt.plot(k_np, S_np[b, ell])
        plt.title(f"Layer {ell}: frequency curves of all samples")
        plt.xlabel("Frequency index k (rfft)")
        plt.ylabel("Spectrum value")
        plt.tight_layout()
        p = f"{out_prefix}_{ell}_curves.png"
        plt.savefig(p, dpi=150)
        plt.show()
        paths.append(p)
    return paths

def plot_layer_heatmaps(
    S_blf: torch.Tensor,          # [B,L,F] real-valued spectra to plot
    # k: torch.Tensor,              # [F] frequency index
    normalize: Optional[Literal["none","prob","zscore","minmax"]] = "none",
    log1p: bool = False,          # apply log1p to enhance contrast
    out_prefix: str = "/mnt/data/layer_heatmap",
) -> List[str]:
    """
    For each layer ℓ, plot a heatmap with rows=samples (B), cols=frequency bins (F).
    normalize:
      - 'none'  : raw values
      - 'prob'  : per (B, L) row-probability along k (sum=1)
      - 'zscore': per (B, L) (x-mean)/std along k
      - 'minmax': per (B, L) (x-min)/(max-min)
    """
    B, L, F = S_blf.shape
    S = S_blf.clone()
    if normalize == "prob":
        S = S / (S.sum(dim=-1, keepdim=True) + 1e-8)
    elif normalize == "zscore":
        mu = S.mean(dim=-1, keepdim=True)
        sd = S.std(dim=-1, keepdim=True).clamp_min(1e-8)
        S = (S - mu) / sd
    elif normalize == "minmax":
        mn = S.amin(dim=-1, keepdim=True)
        mx = S.amax(dim=-1, keepdim=True)
        S = (S - mn) / (mx - mn + 1e-8)

    if log1p:
        S = torch.log1p(S.clamp_min(0))

    paths = []
    S_np = S.cpu().numpy()

    for ell in range(L):
        plt.figure()
        plt.imshow(S_np[:, ell, :], aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(f"Layer {ell}: spectrum heatmap (rows=samples, cols=frequency)")
        plt.xlabel("Frequency index k (rfft)")
        plt.ylabel("Sample index")
        plt.tight_layout()
        p = f"{out_prefix}_{ell}.png"
        plt.savefig(p, dpi=150)
        plt.show()
        paths.append(p)
    return paths

def plot_heatmap(data_2d: np.ndarray, title: str, xlabel: str, ylabel: str, outfile: str):
    plt.figure()
    plt.imshow(data_2d, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile),exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.show()

def plot_curve(xa: np.ndarray, ya: np.ndarray, title: str, xlabel: str, ylabel: str, outfile: str):
    plt.figure()
    plt.plot(xa, ya)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.show()

def plot_bar(xa: np.ndarray, ya: np.ndarray, title: str, xlabel: str, ylabel: str, outfile: str):
    plt.figure()
    plt.bar(xa, ya)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.show()

def make_synthetic_x(B=8, D=256, L=6, noise_std=0.3, seed=0):
    torch.manual_seed(seed)
    base_k = torch.linspace(4, 36, steps=L).round().to(torch.int64)  # [L]
    base_k = base_k.clamp(1, D//2)
    x = torch.zeros(B, D, L, dtype=torch.float32)
    n = torch.arange(D, dtype=torch.float32)
    for ell in range(L):
        k0 = int(base_k[ell].item())
        phase = 2*math.pi*torch.rand(B)
        amp = 0.5 + torch.rand(B)
        # Broadcasted cosine per batch
        signal = torch.cos(2.0*math.pi*k0*n[None, :]/D + phase[:, None]) * amp[:, None]  # [B,D]
        x[:, :, ell] = signal + noise_std*torch.randn(B, D)
    return x


def amplify_vertical_variation_mean_preserving(M, alpha=100.0, eps=1e-6, clip_std=None):
    """
    M: [n_samples, n_cells] 的矩阵（你的热图数据）
    alpha: 纵向对比度放大倍数
    clip_std: 可选，放大前先把每列的 z 值截断到 [-clip_std, clip_std]，如 3
    """
    mu = M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, keepdims=True)
    
    M_new = alpha * (M-mu)
    return M_new





# 序列维频谱细化

def rotated_grid_fft_spectrum_samples(x: torch.Tensor,
                                      K: int,
                                      seq_dim: int = 1,
                                      norm: str = "ortho"):
    """
    旋转频率采样格点的 FFT（每次旋转 Δθ = 2π/K）。
    仅沿 hidden_dim 聚合，保留每个 sample 的幅值。

    参数
    ----
    x: [n_sample, seq_len, hidden_dim]（或通过 seq_dim 指定序列维）
    K: 旋转次数
    seq_dim: 沿该维做 FFT
    norm: FFT 归一化（'backward' | 'ortho' | 'forward'）

    返回
    ----
    angles_sorted: [K*N] 角度（已排序到 [0,2π)）
    per_sample_vals_sorted: [K*N, n_sample] —— 每个角度处每个 sample 的幅值（已在 hidden_dim 上平均）
    """
    assert x.ndim == 3, "x must be [n_sample, seq_len, hidden_dim]"
    # 调整到 [B, T, D]
    if seq_dim != 1:
        x = x.permute(0, seq_dim, *[d for d in range(x.ndim) if d not in (0, seq_dim)])
    x = x.float()
    B, T, D = x.shape

    n = torch.arange(T, device=x.device, dtype=x.dtype).view(1, T, 1)    # [1,T,1]
    k = torch.arange(T, device=x.device)
    theta_k = 2.0 * torch.pi * k / T                                     # [T]

    angles_list = []
    per_sample_vals_list = []   # 每个旋转得到 [T, B]

    for r in range(K):
        beta = r / K
        # 时域调制：x[n] * exp(-j 2π beta n)
        phase = -2.0 * torch.pi * beta * n                               # [1,T,1]
        mod = torch.complex(torch.cos(phase), torch.sin(phase))          # [1,T,1]
        x_mod = x.to(torch.complex64) * mod                              # [B,T,D]

        # FFT
        X = torch.fft.fft(x_mod, dim=1, norm=norm)                       # [B,T,D]
        mag = X.abs().mean(dim=2)                                        # [B,T] 仅沿 hidden_dim 聚合

        # 本次旋转的角度位置
        theta = (theta_k + 2.0 * np.pi * r / K) % (2.0 * np.pi)          # [T]

        angles_list.append(theta.detach().cpu().numpy())                 # [T]
        per_sample_vals_list.append(mag.detach().cpu().numpy().T)        # [T,B]
    # ipdb.set_trace()
    # 拼接并按角度排序
    angles = np.concatenate(angles_list, axis=0)                         # [K*T]
    vals_TB = np.concatenate(per_sample_vals_list, axis=0)               # [K*T, B]
    order = np.argsort(angles)
    angles_sorted = angles[order]
    per_sample_vals_sorted = vals_TB[order, :]                           # [K*T, B]
    return angles_sorted, per_sample_vals_sorted

def seaborn_plot_mean_sd_band(angles_sorted: np.ndarray,
                              per_sample_vals_sorted: np.ndarray,
                              out_path: str = "rot_grid_fft_sns.png"):
    """
    用 seaborn.lineplot 绘制均值 ± 标准差带。
    """
    # 组装 tidy DataFrame: 每行 = 一个 (angle, sample, value)
    KTN, B = per_sample_vals_sorted.shape
    df = pd.DataFrame({
        "angle": np.repeat(angles_sorted, B),
        "value": per_sample_vals_sorted.reshape(-1),
        "sample": np.tile(np.arange(B), KTN)
    })

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 4.5))
    # seaborn >= 0.12: errorbar='sd'；若是旧版可改用 ci='sd'
    ax = sns.lineplot(data=df, x="angle", y="value", errorbar="sd", estimator="mean")
    ax.set_xlim(0, 2*np.pi)
    ax.set_xlabel("Angle (radians)")
    ax.set_ylabel("Amplitude (mean ± 1 SD across samples)")
    ax.set_title("Rotated Frequency-Sampling Grid FFT (hidden-dim aggregated)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _aggregate_prob_to_bands(prob_lf, k_bands, base, band_mode):
    """Aggregate [L, F] layer-frequency probabilities into k prism bands (mean over layers)."""
    prob = prob_lf.float() if isinstance(prob_lf, torch.Tensor) else torch.tensor(prob_lf, dtype=torch.float32)
    if prob.ndim == 1:
        prob = prob.unsqueeze(0)
    mean_prob = prob.mean(dim=0)
    F = mean_prob.shape[0]
    _, _, starts, ends = allocate_prism_bands(F, k=k_bands, base=base, mode=band_mode, device=mean_prob.device)
    band_vals = []
    for bi in range(k_bands):
        s, e = int(starts[bi].item()), int(ends[bi].item())
        band_vals.append(mean_prob[s:e].sum().item())
    return band_vals, starts, ends


def compute_band_energy_concentration(
    prob_per_layer,
    k_bands: int = 8,
    base: int = 4,
    band_mode: str = "uniform",
) -> pd.DataFrame:
    prob = prob_per_layer.float() if isinstance(prob_per_layer, torch.Tensor) else torch.tensor(prob_per_layer, dtype=torch.float32)
    L, F = prob.shape
    _, _, starts, ends = allocate_prism_bands(F, k=k_bands, base=base, mode=band_mode, device=prob.device)
    rows = []
    for ell in range(L):
        layer_prob = prob[ell]
        band_shares = []
        for bi in range(k_bands):
            s, e = int(starts[bi].item()), int(ends[bi].item())
            band_shares.append(layer_prob[s:e].sum().item())
        total = sum(band_shares) + 1e-8
        cum = 0.0
        for bi, share in enumerate(band_shares):
            norm_share = share / total
            cum += norm_share
            rows.append({
                "layer": ell,
                "band": bi,
                "band_start": int(starts[bi].item()),
                "band_end": int(ends[bi].item()),
                "energy_share": norm_share,
                "cumulative_share": cum,
            })
    return pd.DataFrame(rows)


def plot_psd_band_distribution(
    prob_per_layer,
    k_bands: int = 8,
    base: int = 4,
    band_mode: str = "uniform",
    title: str = "PSD Energy Distribution",
    out_png: str = "psd_band_distribution.png",
    out_csv: str = "psd_band_energy.csv",
    highlight_band_range=None,
):
    band_vals, starts, ends = _aggregate_prob_to_bands(prob_per_layer, k_bands, base, band_mode)
    total = sum(band_vals) + 1e-8
    rows = []
    for bi in range(k_bands):
        rows.append({
            "band": bi,
            "band_start": int(starts[bi].item()),
            "band_end": int(ends[bi].item()),
            "energy_share": band_vals[bi] / total,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    hi_end = highlight_band_range[1] if highlight_band_range else 0
    colors = [
        "darkorange" if highlight_band_range and highlight_band_range[0] <= bi < hi_end else "steelblue"
        for bi in range(k_bands)
    ]
    plt.figure(figsize=(9, 5))
    x = np.arange(k_bands)
    plt.bar(x, [band_vals[bi] / total for bi in range(k_bands)], color=colors, alpha=0.85)
    plt.xticks(x, [f"B{bi}" for bi in range(k_bands)])
    plt.xlabel("Frequency band")
    plt.ylabel("Mean energy share")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    return df


def plot_psd_cumulative_curve(
    prob_per_layer,
    k_bands: int = 8,
    base: int = 4,
    band_mode: str = "uniform",
    threshold: float = 0.9,
    title: str = "Cumulative PSD Energy",
    out_png: str = "psd_cumulative_curve.png",
    highlight_band_range=None,
):
    band_vals, _, _ = _aggregate_prob_to_bands(prob_per_layer, k_bands, base, band_mode)
    total = sum(band_vals) + 1e-8
    shares = np.array([v / total for v in band_vals])
    cum = np.cumsum(shares)

    plt.figure(figsize=(8, 5))
    plt.plot(range(k_bands), cum, marker="o", linewidth=2, color="darkorange")
    plt.axhline(threshold, color="red", linestyle="--", label=f"{threshold:.0%} threshold")
    if highlight_band_range:
        lo, hi = highlight_band_range
        plt.axvspan(lo - 0.5, hi - 0.5, alpha=0.1, color="red", label=f"Bands [{lo},{hi})")
    plt.xlabel("Frequency band index")
    plt.ylabel("Cumulative energy share")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


if __name__=='__main__':
    demo_x = make_synthetic_x(B=8, D=256, L=6, noise_std=0.25, seed=42)
    stats = fft_layer_freq_distribution(demo_x, use_rfft=True, measure="mag", topk=3)

    L = int(stats["L"].item())
    F = int(stats["F"].item())
    D = int(stats["D"].item())
    k = stats["k"].cpu().numpy()
    mean_spec = stats["mean_spectrum"].cpu().numpy()    # [L,F]
    prob = stats["prob_per_layer"].cpu().numpy()        # [L,F]
    dom = stats["dominant_counts"].cpu().numpy()        # [L,F]

    plot_heatmap(
        data_2d=mean_spec,
        title="Per-layer mean spectrum (|X|, averaged over batch)",
        xlabel="Frequency index k (rfft)",
        ylabel="Layer index (0-based)",
        outfile="/mnt/data/mean_spectrum_heatmap.png",
    )

    plot_heatmap(
        data_2d=prob,
        title="Per-layer normalized spectrum (probability over frequency)",
        xlabel="Frequency index k (rfft)",
        ylabel="Layer index (0-based)",
        outfile="/mnt/data/prob_per_layer_heatmap.png",
    )

    layers_to_show = list(range(min(3, L)))
    for ell in layers_to_show:
        plot_curve(
            xa=k, ya=mean_spec[ell],
            title=f"Layer {ell}: mean spectrum vs k",
            xlabel="Frequency index k (rfft)",
            ylabel="Mean |X|",
            outfile=f"/mnt/data/layer_{ell}_mean_spectrum.png",
        )
        plot_bar(
            xa=k, ya=dom[ell],
            title=f"Layer {ell}: dominant frequency counts (Top-1 across batch)",
            xlabel="Frequency index k (rfft)",
            ylabel="Counts",
            outfile=f"/mnt/data/layer_{ell}_dominant_counts.png",
        )
