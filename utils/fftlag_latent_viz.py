"""
Training-aligned visualizations for FFT-LAG latent analysis (Exp4).

Training path (V1): latent_out -> mean over queries -> freq_gate -> broadcast gate on all F bins.
These utilities emphasize gate_input / latent_out geometry and value-weighted spectral readout,
not raw per-query attention mass alone.
"""

from __future__ import annotations

import os
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

GateInputMode = Literal["mean", "concat"]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from AMPCliff.spectrual_filter.filter import allocate_prism_bands
from AMPCliff.utils.std_logger import Logger

TensorLike = Union[torch.Tensor, np.ndarray]


def sample_attn_matrix(attn_weights: torch.Tensor) -> torch.Tensor:
    """Normalize to (num_queries, num_freq) for one sample."""
    w = attn_weights
    if w.dim() == 4:
        w = w.mean(dim=1)
    if w.dim() == 3:
        if w.shape[0] == 1:
            w = w[0]
        else:
            raise ValueError(f"Expected per-sample attn (L,F) or (1,L,F), got {tuple(w.shape)}")
    if w.dim() != 2:
        raise ValueError(f"Expected rank-2 attn matrix, got {tuple(w.shape)}")
    return w


def attn_matrix_to_wide_df(attn_weights: TensorLike) -> pd.DataFrame:
    """Convert head-averaged (query, freq) attention to wide CSV layout."""
    if isinstance(attn_weights, torch.Tensor):
        mat = sample_attn_matrix(attn_weights).numpy()
    else:
        mat = sample_attn_matrix(torch.as_tensor(attn_weights)).numpy()
    num_queries, num_freq = mat.shape
    freq_cols = [f"freq_{i}" for i in range(num_freq)]
    df = pd.DataFrame(mat, columns=freq_cols)
    df.insert(0, "query", np.arange(num_queries, dtype=int))
    return df


def export_attn_matrix_wide_csv(attn_weights: TensorLike, out_csv: str) -> pd.DataFrame:
    """Write head-averaged attention matrix as wide CSV (query x freq bins)."""
    df = attn_matrix_to_wide_df(attn_weights)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    Logger.info(f"[saved] {out_csv}")
    return df


def expected_band_mass(
    num_freq: int,
    k_bands: int = 8,
    base: int = 4,
    band_mode: str = "uniform",
) -> np.ndarray:
    """Uniform attention: mass per band equals band_width / num_freq (sums to 1)."""
    _, sizes, _, _ = allocate_prism_bands(num_freq, k=k_bands, base=base, mode=band_mode)
    widths = sizes.numpy().astype(np.float64)
    return widths / float(num_freq)


def _band_slices(num_freq: int, k_bands: int, base: int, band_mode: str):
    _, _, starts, ends = allocate_prism_bands(num_freq, k=k_bands, base=base, mode=band_mode)
    return starts, ends


def freq_bin_energy(freq_tokens: torch.Tensor) -> torch.Tensor:
    """Per-bin energy (F,) from freq_tokens (F, 2D) or (B, F, 2D)."""
    if freq_tokens.dim() == 3:
        return (freq_tokens ** 2).sum(dim=-1).mean(dim=0)
    if freq_tokens.dim() == 2:
        return (freq_tokens ** 2).sum(dim=-1)
    raise ValueError(f"freq_tokens must be (F,2D) or (B,F,2D), got {tuple(freq_tokens.shape)}")


def compute_query_band_mass(
    attn_weights: torch.Tensor,
    k_bands: int,
    base: int,
    band_mode: str,
) -> pd.DataFrame:
    """query x band raw attention mass for one sample."""
    w = sample_attn_matrix(attn_weights)
    num_queries, num_freq = w.shape
    starts, ends = _band_slices(num_freq, k_bands, base, band_mode)
    rows = []
    for qi in range(num_queries):
        for bi in range(k_bands):
            s = int(starts[bi].item())
            e = int(ends[bi].item())
            mass = w[qi, s:e].sum().item()
            rows.append({
                "query": qi,
                "band": bi,
                "band_start": s,
                "band_end": e,
                "attention_mass": mass,
            })
    return pd.DataFrame(rows)


def compute_attn_band_deviation(
    attn_weights: torch.Tensor,
    k_bands: int,
    base: int,
    band_mode: str,
) -> pd.DataFrame:
    """query x band deviation from uniform-expected mass."""
    mass_df = compute_query_band_mass(attn_weights, k_bands, base, band_mode)
    num_freq = int(mass_df["band_end"].max())
    expected = expected_band_mass(num_freq, k_bands, base, band_mode)
    mass_df = mass_df.copy()
    mass_df["expected_mass"] = mass_df["band"].map(lambda b: expected[int(b)])
    mass_df["deviation"] = mass_df["attention_mass"] - mass_df["expected_mass"]
    return mass_df


def compute_weighted_band_readout(
    attn_weights: torch.Tensor,
    freq_tokens: torch.Tensor,
    k_bands: int,
    base: int,
    band_mode: str,
) -> pd.DataFrame:
    """
    Per query: sum_f attn[q,f] * energy(f), aggregated by band.
    Reflects what spectral content each query actually pulls (attention x value).
    """
    w = sample_attn_matrix(attn_weights)
    energy = freq_bin_energy(freq_tokens).to(dtype=w.dtype, device=w.device)
    num_queries, num_freq = w.shape
    if energy.shape[0] != num_freq:
        raise ValueError(f"energy F={energy.shape[0]} != attn F={num_freq}")

    weighted = w * energy.unsqueeze(0)
    starts, ends = _band_slices(num_freq, k_bands, base, band_mode)
    rows = []
    for qi in range(num_queries):
        for bi in range(k_bands):
            s = int(starts[bi].item())
            e = int(ends[bi].item())
            val = weighted[qi, s:e].sum().item()
            rows.append({
                "query": qi,
                "band": bi,
                "band_start": s,
                "band_end": e,
                "weighted_mass": val,
            })
    return pd.DataFrame(rows)


def _vector_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


def gate_input_mode_from_pooling(pooling: str) -> GateInputMode:
    """Map pooling name to gate_input aggregation used in training."""
    name = str(pooling).strip()
    if name in (
        "fft_latent_attn_gate_v3",
        "fft_latent_attn_gate_v3_1",
        "fft_latent_attn_gate_v3_2",
        "fft_latent_attn_gate_v3_3",
    ):
        return "concat"
    return "mean"


def gate_input_from_latent_out(
    latent_out: torch.Tensor,
    mode: GateInputMode = "mean",
) -> torch.Tensor:
    """
    (L, 2D) or (B, L, 2D) -> gate_input aligned with training.

    mean: FFTLatentAttentionGatePooling (V1)
    concat: FFTLatentAttentionGateV3Pooling
    """
    if mode not in ("mean", "concat"):
        raise ValueError(f"gate_input mode must be 'mean' or 'concat', got {mode!r}")
    if latent_out.dim() == 2:
        if mode == "concat":
            return latent_out.reshape(-1)
        return latent_out.mean(dim=0)
    if latent_out.dim() == 3:
        if mode == "concat":
            return latent_out.reshape(latent_out.shape[0], -1)
        return latent_out.mean(dim=1)
    raise ValueError(f"latent_out must be (L,2D) or (B,L,2D), got {tuple(latent_out.shape)}")


def gate_input_from_freq_tokens(freq_tokens: torch.Tensor) -> torch.Tensor:
    """(F, 2D) or (B, F, 2D) -> gate_input from freq_tokens.mean(dim=1) (fft_gate_only path)."""
    if freq_tokens.dim() == 3:
        if freq_tokens.shape[0] != 1:
            raise ValueError("Batch freq_tokens not supported; pass single sample (F,2D)")
        freq_tokens = freq_tokens[0]
    if freq_tokens.dim() != 2:
        raise ValueError(f"Expected (F,2D), got {tuple(freq_tokens.shape)}")
    return freq_tokens.mean(dim=0)


def apply_freq_gate_mlp(freq_gate: nn.Module, gate_input: torch.Tensor) -> torch.Tensor:
    """sigmoid(freq_gate(gate_input)) -> (2D,) for one sample."""
    gi = gate_input.float()
    if gi.dim() == 1:
        gi = gi.unsqueeze(0)
    device = next(freq_gate.parameters()).device
    gi = gi.to(device)
    with torch.no_grad():
        return torch.sigmoid(freq_gate(gi)).squeeze(0).cpu()


def compute_gate_input_compare(
    latent_out: torch.Tensor,
    freq_tokens: torch.Tensor,
    freq_gate: nn.Module,
    raw_gate: Optional[torch.Tensor] = None,
    mode: GateInputMode = "mean",
) -> Dict[str, float]:
    """
    Compare training gate_input (latent path) vs freq uniform pool (mean over F).
    Counterfactual: same freq_gate MLP on both inputs.
    """
    if latent_out.dim() == 3:
        if latent_out.shape[0] != 1:
            raise ValueError("Batch latent_out not supported")
        latent_out = latent_out[0]
    if freq_tokens.dim() == 3:
        if freq_tokens.shape[0] != 1:
            raise ValueError("Batch freq_tokens not supported")
        freq_tokens = freq_tokens[0]

    gi_latent = gate_input_from_latent_out(latent_out, mode=mode)
    gi_freq = gate_input_from_freq_tokens(freq_tokens)
    if mode == "concat":
        num_latents = int(latent_out.shape[0])
        gi_freq_for_gate = gi_freq.unsqueeze(0).expand(num_latents, -1).reshape(-1)
    else:
        gi_freq_for_gate = gi_freq

    gate_latent = apply_freq_gate_mlp(freq_gate, gi_latent)
    gate_freq_cf = apply_freq_gate_mlp(freq_gate, gi_freq_for_gate)

    out: Dict[str, float] = {
        "cos_latent_freq": _vector_cosine(gi_latent, gi_freq_for_gate),
        "gi_latent_l2": float(gi_latent.norm().item()),
        "gi_freq_l2": float(gi_freq.norm().item()),
        "cos_gate_latent_vs_freq_cf": _vector_cosine(gate_latent, gate_freq_cf),
    }
    if raw_gate is not None:
        rg = raw_gate.float()
        if rg.dim() == 2:
            rg = rg.squeeze(0)
        out["cos_gate_latent_vs_cached_raw"] = _vector_cosine(gate_latent, rg)
    return out


def compute_gate_input_band_profiles(
    attn_weights: torch.Tensor,
    freq_tokens: torch.Tensor,
    k_bands: int,
    base: int,
    band_mode: str,
) -> pd.DataFrame:
    """
    1x8 band profiles before gate: latent path (mean over queries of weighted readout)
    vs freq uniform pool (sum energy in band / F).
    """
    readout_df = compute_weighted_band_readout(
        attn_weights, freq_tokens, k_bands, base, band_mode
    )
    latent_pooled = readout_df.groupby("band", as_index=False).agg(
        latent_pooled_readout=("weighted_mass", "mean"),
        band_start=("band_start", "first"),
        band_end=("band_end", "first"),
    )

    energy = freq_bin_energy(freq_tokens)
    num_freq = int(energy.shape[0])
    starts, ends = _band_slices(num_freq, k_bands, base, band_mode)
    freq_rows = []
    for bi in range(k_bands):
        s = int(starts[bi].item())
        e = int(ends[bi].item())
        freq_rows.append({
            "band": bi,
            "band_start": s,
            "band_end": e,
            "freq_uniform_pool_energy": float(energy[s:e].sum().item() / num_freq),
        })
    freq_df = pd.DataFrame(freq_rows)
    return latent_pooled.merge(freq_df, on=["band", "band_start", "band_end"], how="inner")


def compute_latent_out_contribution(
    latent_out: torch.Tensor,
    mode: GateInputMode = "mean",
) -> pd.DataFrame:
    """
    Per-query contribution relative to the training gate_input path.

    mean: deviation / cosine vs mean(latent_out) (V1).
    concat: deviation / cosine vs mean(latent_out) as reference (V3 uses full concat for MLP).
    Columns: query, l2_deviation, cos_to_gate_input
    """
    if latent_out.dim() == 3:
        if latent_out.shape[0] != 1:
            raise ValueError("Batch latent_out not supported; pass single sample (L,2D)")
        latent_out = latent_out[0]
    if latent_out.dim() != 2:
        raise ValueError(f"Expected (L,2D), got {tuple(latent_out.shape)}")

    lo = latent_out.float()
    ref = lo.mean(dim=0)
    ref_norm = ref.norm() + 1e-8
    rows = []
    for qi in range(lo.shape[0]):
        v = lo[qi]
        l2_dev = (v - ref).norm().item()
        cos_gi = float(torch.dot(v, ref) / (v.norm() * ref_norm + 1e-8))
        rows.append({
            "query": qi,
            "l2_deviation": l2_dev,
            "cos_to_gate_input": cos_gi,
        })
    return pd.DataFrame(rows)


def latent_out_cosine_matrix(latent_out: torch.Tensor) -> np.ndarray:
    """(L, L) cosine similarity between latent query vectors."""
    if latent_out.dim() == 3:
        if latent_out.shape[0] != 1:
            raise ValueError("Batch latent_out not supported")
        latent_out = latent_out[0]
    lo = latent_out.float()
    L = lo.shape[0]
    mat = np.eye(L, dtype=np.float64)
    for i in range(L):
        for j in range(i + 1, L):
            vi, vj = lo[i], lo[j]
            c = float(torch.dot(vi, vj) / (vi.norm() * vj.norm() + 1e-8))
            mat[i, j] = c
            mat[j, i] = c
    return mat


def aggregate_latent_out_cosine(samples: Sequence[torch.Tensor]) -> np.ndarray:
    """Mean LxL cosine across samples."""
    if not samples:
        raise ValueError("No samples for cosine aggregation")
    acc = np.zeros_like(latent_out_cosine_matrix(samples[0]), dtype=np.float64)
    for lo in samples:
        acc += latent_out_cosine_matrix(lo)
    return acc / len(samples)


def mean_latent_out_offdiag_cosine(latent_out: torch.Tensor) -> Tuple[float, float]:
    """Return (mean, max) off-diagonal cosine similarity between latent query vectors."""
    mat = latent_out_cosine_matrix(latent_out)
    L = mat.shape[0]
    if L < 2:
        return 0.0, 0.0
    off = mat[~np.eye(L, dtype=bool)]
    return float(off.mean()), float(off.max())


def summarize_latent_query_discriminability(
    latent_out: torch.Tensor,
    attn_weights: torch.Tensor,
    idx: int,
    k_bands: int,
    base: int,
    band_mode: str,
    mode: GateInputMode = "mean",
) -> Dict[str, float]:
    """Scalars for whether latent queries remain hard to distinguish."""
    if latent_out.dim() == 3:
        if latent_out.shape[0] != 1:
            raise ValueError("Batch latent_out not supported")
        latent_out = latent_out[0]

    mean_off, max_off = mean_latent_out_offdiag_cosine(latent_out)
    contrib = compute_latent_out_contribution(latent_out, mode=mode)
    div = compute_query_diversity(attn_weights, idx, k_bands, base, band_mode)

    out: Dict[str, float] = {
        "idx": int(idx),
        "mean_latent_out_offdiag_cosine": mean_off,
        "max_latent_out_offdiag_cosine": max_off,
        "mean_cos_to_gate_input": float(contrib["cos_to_gate_input"].mean()),
        "mean_query_cosine_distance": float(div["mean_query_cosine_distance"]),
        "mean_query_js_divergence": float(div["mean_query_js_divergence"]),
        "query_attn_std_mean": float(div["query_attn_std_mean"]),
        "num_queries": float(div["num_queries"]),
    }
    return out


def compute_gate_input_stats(
    latent_out: torch.Tensor,
    raw_gate: Optional[torch.Tensor] = None,
    mode: GateInputMode = "mean",
) -> Dict[str, float]:
    """Per-sample scalars linking latent_out to gate_input (mean or concat)."""
    gate_input = gate_input_from_latent_out(latent_out, mode=mode)
    if gate_input.dim() == 2:
        gate_input = gate_input[0]
    stats = {
        "gate_input_l2": float(gate_input.norm().item()),
        "gate_input_mean": float(gate_input.mean().item()),
        "gate_input_std": float(gate_input.std().item()),
    }
    if raw_gate is not None:
        rg = raw_gate.float()
        if rg.dim() == 2:
            rg = rg[0]
        stats["raw_gate_mean"] = float(rg.mean().item())
        stats["raw_gate_std"] = float(rg.std().item())
        stats["raw_gate_l2"] = float(rg.norm().item())
    return stats


def compute_query_diversity(
    attn_weights: torch.Tensor,
    idx: int,
    k_bands: int,
    base: int,
    band_mode: str,
) -> Dict[str, float]:
    """Legacy band-profile diversity on attention (kept for CSV compatibility)."""
    w = sample_attn_matrix(attn_weights)
    num_queries = w.shape[0]
    starts, ends = _band_slices(w.shape[1], k_bands, base, band_mode)

    band_profiles = []
    for qi in range(num_queries):
        prof = []
        for bi in range(k_bands):
            s = int(starts[bi].item())
            e = int(ends[bi].item())
            prof.append(w[qi, s:e].sum().item())
        band_profiles.append(np.array(prof, dtype=np.float64))
    band_profiles = np.stack(band_profiles, axis=0)
    band_profiles = band_profiles / (band_profiles.sum(axis=1, keepdims=True) + 1e-8)

    cos_dists = []
    js_dists = []
    for i in range(num_queries):
        for j in range(i + 1, num_queries):
            vi = band_profiles[i]
            vj = band_profiles[j]
            cos_dists.append(
                1.0 - float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-8))
            )
            m = 0.5 * (vi + vj)
            js = 0.5 * (
                np.sum(vi * np.log((vi + 1e-8) / (m + 1e-8)))
                + np.sum(vj * np.log((vj + 1e-8) / (m + 1e-8)))
            )
            js_dists.append(float(js))

    return {
        "idx": idx,
        "num_queries": num_queries,
        "mean_query_cosine_distance": float(np.mean(cos_dists)) if cos_dists else 0.0,
        "mean_query_js_divergence": float(np.mean(js_dists)) if js_dists else 0.0,
        "query_attn_std_mean": float(w.std(dim=1).mean().item()),
    }


def filter_by_idx(df: Optional[pd.DataFrame], idx: int) -> Optional[pd.DataFrame]:
    """Return rows for one peptide; None/empty in, None/empty out."""
    if df is None or df.empty:
        return df
    if "idx" not in df.columns:
        return df
    out = df[df["idx"] == idx]
    return out.copy() if not out.empty else out


def collect_sample_idxs(
    contrib_df: Optional[pd.DataFrame],
    dev_df: Optional[pd.DataFrame],
    readout_df: Optional[pd.DataFrame],
    compare_df: Optional[pd.DataFrame] = None,
    band_profile_df: Optional[pd.DataFrame] = None,
    latent_out_by_idx: Optional[Dict[int, torch.Tensor]] = None,
) -> List[int]:
    """Peptide indices that have rows in all provided frames and latent_out."""
    idx_sets: List[set] = []
    for df in (contrib_df, dev_df, readout_df, compare_df, band_profile_df):
        if df is not None and not df.empty and "idx" in df.columns:
            idx_sets.append(set(int(x) for x in df["idx"].unique()))
    if latent_out_by_idx:
        idx_sets.append(set(int(k) for k in latent_out_by_idx.keys()))
    if not idx_sets:
        return []
    common = idx_sets[0]
    for s in idx_sets[1:]:
        common &= s
    return sorted(common)


def _pivot_query_band(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    work = df
    if "idx" in work.columns:
        n_idx = work["idx"].nunique()
        if n_idx > 1:
            raise ValueError(
                f"_pivot_query_band expects one peptide (idx), got {n_idx} distinct idx values"
            )
        work = work.drop(columns=["idx"])
    return work.groupby(["query", "band"])[value_col].mean().unstack("band")


def plot_query_band_heatmap(
    df_mass: pd.DataFrame,
    out_png: str,
    value_col: str = "attention_mass",
    title: str = "Latent query band mass",
):
    """Legacy raw attention mass heatmap."""
    pivot = _pivot_query_band(df_mass, value_col)
    plt.figure(figsize=(10, max(4, pivot.shape[0] * 0.35)))
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".3f")
    plt.xlabel("Frequency band")
    plt.ylabel("Latent query")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    Logger.info(f"[saved] {out_png}")


def plot_raw_attn_heatmap(
    attn_weights: TensorLike,
    out_png: str,
    title: str = "Cross-attention score (head-averaged)",
    annot: bool = False,
    out_csv: Optional[str] = None,
):
    """Heatmap of head-averaged latent query x frequency-bin attention scores."""
    w = sample_attn_matrix(
        attn_weights if isinstance(attn_weights, torch.Tensor) else torch.as_tensor(attn_weights)
    )
    mat = w.numpy()
    num_queries, num_freq = mat.shape
    fig_w = max(10, min(24, num_freq * 0.35))
    fig_h = max(4, num_queries * 0.35)
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(mat, cmap="viridis", annot=annot, fmt=".3f", cbar_kws={"label": "attention"})
    plt.xlabel("Frequency bin")
    plt.ylabel("Latent query")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    Logger.info(f"[saved] {out_png}")
    if out_csv is not None:
        export_attn_matrix_wide_csv(w, out_csv)


def plot_query_contribution(
    contrib_df: pd.DataFrame,
    out_png: str,
    title: str = "Latent query contribution to gate_input",
):
    """Bar chart: L2 deviation from mean(latent_out) per query (one peptide)."""
    work = contrib_df
    if "idx" in work.columns:
        if work["idx"].nunique() > 1:
            raise ValueError("plot_query_contribution expects one idx per call")
        work = work.drop(columns=["idx"])
    agg = work.groupby("query").agg(
        l2_mean=("l2_deviation", "mean"),
        l2_std=("l2_deviation", "std"),
    ).reset_index()
    agg["l2_std"] = agg["l2_std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = agg["query"].astype(int)
    ax.bar(x, agg["l2_mean"], yerr=agg["l2_std"], capsize=3, color="steelblue", alpha=0.85)
    ax.set_xlabel("Latent query")
    ax.set_ylabel("L2 deviation from mean(latent_out)")
    ax.set_title(title)
    ax.set_xticks(x)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    Logger.info(f"[saved] {out_png}")


def plot_latent_out_cosine_heatmap(
    cos_mat: np.ndarray,
    out_png: str,
    title: str = "Latent out cosine similarity (gate input space)",
):
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cos_mat,
        cmap="RdYlBu_r",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".3f",
        square=True,
    )
    plt.xlabel("Latent query")
    plt.ylabel("Latent query")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    Logger.info(f"[saved] {out_png}")


def plot_band_heatmap_from_df(
    df: pd.DataFrame,
    value_col: str,
    out_png: str,
    title: str,
    cmap: str = "coolwarm",
    center: Optional[float] = 0.0,
    fmt: str = ".4f",
):
    pivot = _pivot_query_band(df, value_col)
    plt.figure(figsize=(10, max(4, pivot.shape[0] * 0.35)))
    heatmap_kw = {"cmap": cmap, "annot": True, "fmt": fmt}
    if center is not None:
        heatmap_kw["center"] = center
    sns.heatmap(pivot, **heatmap_kw)
    plt.xlabel("Frequency band")
    plt.ylabel("Latent query")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    Logger.info(f"[saved] {out_png}")


def plot_attn_deviation_heatmap(
    dev_df: pd.DataFrame,
    out_png: str,
    title: str = "Attention mass deviation from uniform",
):
    plot_band_heatmap_from_df(
        dev_df,
        value_col="deviation",
        out_png=out_png,
        title=title,
        cmap="coolwarm",
        center=0.0,
        fmt=".4f",
    )


def plot_weighted_readout_heatmap(
    readout_df: pd.DataFrame,
    out_png: str,
    title: str = "Value-weighted band readout per query",
):
    plot_band_heatmap_from_df(
        readout_df,
        value_col="weighted_mass",
        out_png=out_png,
        title=title,
        cmap="viridis",
        center=None,
        fmt=".3f",
    )


def plot_gate_input_latent_vs_freq(
    compare_df: pd.DataFrame,
    out_png: str,
    title: str = "Gate input: latent mean vs freq mean",
):
    """Per-peptide cos(gi_latent, gi_freq) and cos(gate_latent, gate_freq_cf)."""
    n = len(compare_df)
    if n == 0:
        return

    if n == 1:
        row = compare_df.iloc[0]
        fig, axes = plt.subplots(2, 1, figsize=(6, 4))
        x_label = f"idx={int(row['idx'])}"
        axes[0].bar([0], [row["cos_latent_freq"]], color="steelblue", alpha=0.85, width=0.5)
        axes[0].axhline(0.0, color="gray", linewidth=0.8)
        axes[0].set_ylabel("cos(gi_latent, gi_freq)")
        axes[0].set_title(f"{title} — gate_input vectors")
        axes[0].set_xticks([0])
        axes[0].set_xticklabels([x_label])
        axes[0].set_ylim(-0.05, 1.05)

        axes[1].bar(
            [0], [row["cos_gate_latent_vs_freq_cf"]], color="darkorange", alpha=0.85, width=0.5
        )
        axes[1].set_ylabel("cos(gate_latent, gate_freq_cf)")
        axes[1].set_xticks([0])
        axes[1].set_xticklabels([x_label])
        axes[1].set_ylim(-0.05, 1.05)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        x_labels = compare_df["idx"].astype(str).tolist()
        x_pos = np.arange(n)

        axes[0].bar(x_pos, compare_df["cos_latent_freq"], color="steelblue", alpha=0.85)
        axes[0].axhline(0.0, color="gray", linewidth=0.8)
        axes[0].set_ylabel("cos(gi_latent, gi_freq)")
        axes[0].set_title(f"{title} — gate_input vectors")
        axes[0].set_ylim(-0.05, 1.05)

        axes[1].bar(
            x_pos, compare_df["cos_gate_latent_vs_freq_cf"], color="darkorange", alpha=0.85
        )
        axes[1].set_ylabel("cos(gate_latent, gate_freq_cf)")
        axes[1].set_xlabel("Peptide idx")
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(x_labels, rotation=45, ha="right")
        axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    Logger.info(f"[saved] {out_png}")


def plot_gate_input_band_profile(
    band_df: pd.DataFrame,
    out_png: str,
    title: str = "Gate-input band profiles (pooled)",
):
    """2-row heatmap: latent_pooled_readout vs freq_uniform_pool_energy for one peptide."""
    work = band_df
    if "idx" in work.columns:
        if work["idx"].nunique() > 1:
            raise ValueError("plot_gate_input_band_profile expects one idx per call")
        work = work.drop(columns=["idx"])
    agg = work.groupby("band", as_index=False).agg(
        latent_pooled_readout=("latent_pooled_readout", "mean"),
        freq_uniform_pool_energy=("freq_uniform_pool_energy", "mean"),
    )
    mat = agg.set_index("band")[["latent_pooled_readout", "freq_uniform_pool_energy"]].T
    mat.index = ["latent_pooled_readout", "freq_uniform_pool"]

    plt.figure(figsize=(10, 3.5))
    sns.heatmap(mat, cmap="viridis", annot=True, fmt=".1f")
    plt.xlabel("Frequency band")
    plt.ylabel("Source")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    Logger.info(f"[saved] {out_png}")


def plot_gate_input_query_summary(
    contrib_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    out_png: str,
    title: str = "Queries vs pooled gate_input",
    gate_input_mode: GateInputMode = "mean",
):
    """Query cos to gate_input for one peptide + gi/gate cos summary."""
    align_label = (
        "concat(latent_out) MLP input; per-query vs mean(latent_out) reference"
        if gate_input_mode == "concat"
        else "mean(latent_out)"
    )
    work = contrib_df
    if "idx" in work.columns:
        if work["idx"].nunique() > 1:
            raise ValueError("plot_gate_input_query_summary expects one idx per call")
        work = work.drop(columns=["idx"])
    agg = work.groupby("query").agg(
        cos_mean=("cos_to_gate_input", "mean"),
        cos_std=("cos_to_gate_input", "std"),
    ).reset_index()
    agg["cos_std"] = agg["cos_std"].fillna(0.0)

    if len(compare_df) == 1:
        mean_cos_lf = float(compare_df["cos_latent_freq"].iloc[0])
        mean_cos_gate = float(compare_df["cos_gate_latent_vs_freq_cf"].iloc[0])
        n_peptides = 1
        cached_cos = None
        if "cos_gate_latent_vs_cached_raw" in compare_df.columns:
            cached_cos = float(compare_df["cos_gate_latent_vs_cached_raw"].iloc[0])
    else:
        mean_cos_lf = float(compare_df["cos_latent_freq"].mean())
        mean_cos_gate = float(compare_df["cos_gate_latent_vs_freq_cf"].mean())
        n_peptides = len(compare_df)
        cached_cos = None
        if "cos_gate_latent_vs_cached_raw" in compare_df.columns:
            cached_cos = float(compare_df["cos_gate_latent_vs_cached_raw"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = agg["query"].astype(int)
    axes[0].bar(x, agg["cos_mean"], yerr=agg["cos_std"], capsize=3, color="steelblue", alpha=0.85)
    axes[0].set_xlabel("Latent query")
    axes[0].set_ylabel("cos(query, gate_input_latent)")
    axes[0].set_title(f"Per-query alignment ({align_label})")
    axes[0].set_xticks(x)

    axes[1].axis("off")
    summary = (
        f"cos(gi_latent, gi_freq) = {mean_cos_lf:.4f}\n"
        f"cos(gate_latent, gate_freq_cf) = {mean_cos_gate:.4f}\n"
        f"n_peptides = {n_peptides}"
    )
    if cached_cos is not None:
        summary += f"\ncos(gate_latent, cached_raw) = {cached_cos:.4f}"
    axes[1].text(0.1, 0.5, summary, fontsize=12, va="center", family="monospace")
    axes[1].set_title("Gate-input path summary")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    Logger.info(f"[saved] {out_png}")


PRIMARY_PLOT_NAMES = frozenset({
    "query_contribution",
    "latent_out_cosine",
    "attn_deviation",
    "weighted_readout",
    "gate_input_compare",
    "gate_input_band_profile",
    "raw_attn_score",
})

DEFAULT_PRIMARY_PLOTS = (
    "query_contribution",
    "latent_out_cosine",
    "attn_deviation",
    "weighted_readout",
    "gate_input_compare",
    "gate_input_band_profile",
    "raw_attn_score",
)


def render_primary_plots(
    *,
    contrib_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    readout_df: pd.DataFrame,
    cos_mat: np.ndarray,
    dataset_label: str,
    out_dir: str = ".",
    primary_plots: Optional[Sequence[str]] = None,
    compare_df: Optional[pd.DataFrame] = None,
    band_profile_df: Optional[pd.DataFrame] = None,
    attn_matrix: Optional[TensorLike] = None,
    gate_input_mode: GateInputMode = "mean",
) -> None:
    """Render training-aligned figures under out_dir."""
    # 空序列表示"不画任何图"（如 fulltest 批量评估传 primary_plots=[]），仅 None 才回退默认
    plots = set(DEFAULT_PRIMARY_PLOTS) if primary_plots is None else set(primary_plots)
    unknown = plots - PRIMARY_PLOT_NAMES
    if unknown:
        raise ValueError(f"Unknown primary_plots: {unknown}")

    if "query_contribution" in plots:
        plot_query_contribution(
            contrib_df,
            out_png=f"{out_dir}/latent_query_contribution.png",
            title=f"Latent query contribution to gate_input ({dataset_label})",
        )
    if "latent_out_cosine" in plots:
        plot_latent_out_cosine_heatmap(
            cos_mat,
            out_png=f"{out_dir}/latent_out_cosine_heatmap.png",
            title=f"Latent out cosine — gate input space ({dataset_label})",
        )
    if "attn_deviation" in plots:
        plot_attn_deviation_heatmap(
            dev_df,
            out_png=f"{out_dir}/latent_attn_deviation_heatmap.png",
            title=f"Attention deviation from uniform ({dataset_label})",
        )
    if "weighted_readout" in plots:
        plot_weighted_readout_heatmap(
            readout_df,
            out_png=f"{out_dir}/latent_weighted_readout_heatmap.png",
            title=f"Value-weighted band readout ({dataset_label})",
        )
    if "gate_input_compare" in plots:
        if compare_df is None or compare_df.empty:
            Logger.warning("Skipped gate_input_compare: missing compare_df")
        else:
            plot_gate_input_latent_vs_freq(
                compare_df,
                out_png=f"{out_dir}/gate_input_latent_vs_freq.png",
                title=f"Gate input latent vs freq ({dataset_label})",
            )
            if contrib_df is not None and not contrib_df.empty:
                plot_gate_input_query_summary(
                    contrib_df,
                    compare_df,
                    out_png=f"{out_dir}/gate_input_query_to_mean.png",
                    title=f"Queries vs gate_input ({dataset_label})",
                    gate_input_mode=gate_input_mode,
                )
    if "gate_input_band_profile" in plots:
        if band_profile_df is None or band_profile_df.empty:
            Logger.warning("Skipped gate_input_band_profile: missing band_profile_df")
        else:
            plot_gate_input_band_profile(
                band_profile_df,
                out_png=f"{out_dir}/gate_input_band_profile.png",
                title=f"Gate-input band profiles ({dataset_label})",
            )
    if "raw_attn_score" in plots:
        if attn_matrix is None:
            Logger.warning("Skipped raw_attn_score: missing attn_matrix")
        else:
            plot_raw_attn_heatmap(
                attn_matrix,
                out_png=f"{out_dir}/attn_score_raw.png",
                title=f"Cross-attention score ({dataset_label})",
                out_csv=f"{out_dir}/attn_score_raw.csv",
            )


def render_primary_plots_per_sample(
    *,
    contrib_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    readout_df: pd.DataFrame,
    latent_out_by_idx: Dict[int, torch.Tensor],
    dataset_label: str,
    out_base: str = ".",
    primary_plots: Optional[Sequence[str]] = None,
    compare_df: Optional[pd.DataFrame] = None,
    band_profile_df: Optional[pd.DataFrame] = None,
    mass_df: Optional[pd.DataFrame] = None,
    attn_weights_by_idx: Optional[Dict[int, torch.Tensor]] = None,
    plot_legacy: bool = False,
    gate_input_mode: GateInputMode = "mean",
) -> None:
    """Render one full primary plot set per peptide under out_base/per_sample/idx_{idx}/."""
    idxs = collect_sample_idxs(
        contrib_df,
        dev_df,
        readout_df,
        compare_df=compare_df,
        band_profile_df=band_profile_df,
        latent_out_by_idx=latent_out_by_idx,
    )
    if not idxs:
        Logger.warning("Skipped per-sample plots: no common peptide idx")
        return

    for idx in idxs:
        lo = latent_out_by_idx.get(idx)
        if lo is None:
            Logger.warning(f"Skip per-sample plots for idx={idx}: missing latent_out")
            continue

        out_dir = os.path.join(out_base, "per_sample", f"idx_{idx}")
        os.makedirs(out_dir, exist_ok=True)
        sample_label = f"{dataset_label} idx={idx}"
        cos_mat = latent_out_cosine_matrix(lo)

        c_idx = filter_by_idx(contrib_df, idx)
        d_idx = filter_by_idx(dev_df, idx)
        r_idx = filter_by_idx(readout_df, idx)
        cmp_idx = filter_by_idx(compare_df, idx) if compare_df is not None else None
        band_idx = filter_by_idx(band_profile_df, idx) if band_profile_df is not None else None

        if c_idx is None or c_idx.empty or d_idx is None or d_idx.empty or r_idx is None or r_idx.empty:
            Logger.warning(f"Skip per-sample plots for idx={idx}: incomplete tabular data")
            continue

        attn_matrix = None
        if attn_weights_by_idx is not None:
            attn_matrix = attn_weights_by_idx.get(idx)

        render_primary_plots(
            contrib_df=c_idx,
            dev_df=d_idx,
            readout_df=r_idx,
            cos_mat=cos_mat,
            dataset_label=sample_label,
            out_dir=out_dir,
            primary_plots=primary_plots,
            compare_df=cmp_idx,
            band_profile_df=band_idx,
            attn_matrix=attn_matrix,
            gate_input_mode=gate_input_mode,
        )

        if plot_legacy and mass_df is not None:
            m_idx = filter_by_idx(mass_df, idx)
            if m_idx is not None and not m_idx.empty:
                legacy_work = m_idx.drop(columns=["idx"]) if "idx" in m_idx.columns else m_idx
                plot_query_band_heatmap(
                    legacy_work,
                    out_png=f"{out_dir}/query_band_heatmap.png",
                    title=f"Latent query band mass [legacy] ({sample_label})",
                )


def mean_attn_matrices(mats: Sequence[torch.Tensor]) -> torch.Tensor:
    """Mean stack of (num_queries, num_freq) attention matrices."""
    stacked = torch.stack([sample_attn_matrix(m).float() for m in mats], dim=0)
    return stacked.mean(dim=0)


def attn_matrix_to_long_df(
    attn_weights: TensorLike,
    idx: int,
    *,
    pool_queries: bool = True,
) -> pd.DataFrame:
    """
    Melt head-averaged (query, freq) attention to long format.

    pool_queries=True: one row per (idx, freq_bin) with all query scores pooled.
    pool_queries=False: one row per (idx, query, freq_bin).
    """
    mat = sample_attn_matrix(
        attn_weights if isinstance(attn_weights, torch.Tensor) else torch.as_tensor(attn_weights)
    ).numpy()
    num_queries, num_freq = mat.shape
    rows = []
    for fi in range(num_freq):
        if pool_queries:
            for qi in range(num_queries):
                rows.append({
                    "idx": int(idx),
                    "freq_bin": int(fi),
                    "attn_score": float(mat[qi, fi]),
                })
        else:
            for qi in range(num_queries):
                rows.append({
                    "idx": int(idx),
                    "query": int(qi),
                    "freq_bin": int(fi),
                    "attn_score": float(mat[qi, fi]),
                })
    return pd.DataFrame(rows)


def aggregate_attn_by_idx_across_seeds(
    seed_dirs: Sequence[Union[str, os.PathLike]],
    dataset: str,
    *,
    subdir: str = "exp4_latent_fulltest",
    pt_filename: str = "latent_attn_weights.pt",
    min_seeds: int = 1,
) -> Tuple[Dict[int, torch.Tensor], List[str]]:
    """
    Load per-seed latent_attn_weights.pt and return seed-mean attention per idx.

    Returns (mean_attn_by_idx, warnings).
    """
    from collections import defaultdict
    from pathlib import Path

    by_idx: Dict[int, List[torch.Tensor]] = defaultdict(list)
    warnings: List[str] = []

    for seed_dir in seed_dirs:
        seed_path = Path(seed_dir)
        pt_path = seed_path / dataset / subdir / pt_filename
        if not pt_path.is_file():
            warnings.append(f"[MISSING] {pt_path}")
            continue

        payload = torch.load(pt_path, map_location="cpu")
        samples = payload.get("samples") or []
        if not samples:
            warnings.append(f"[EMPTY] {pt_path}")
            continue

        for sample in samples:
            idx = int(sample["idx"])
            by_idx[idx].append(sample_attn_matrix(sample["attn_weights"]))

    mean_by_idx: Dict[int, torch.Tensor] = {}
    for idx, mats in sorted(by_idx.items()):
        if len(mats) < min_seeds:
            warnings.append(
                f"[SKIP] idx={idx}: only {len(mats)} seeds (min_seeds={min_seeds})"
            )
            continue
        shapes = {tuple(m.shape) for m in mats}
        if len(shapes) != 1:
            warnings.append(f"[SHAPE_MISMATCH] idx={idx}: shapes={sorted(shapes)}")
            continue
        mean_by_idx[idx] = mean_attn_matrices(mats)

    return mean_by_idx, warnings


def build_pooled_freq_distribution_df(
    mean_attn_by_idx: Dict[int, torch.Tensor],
    *,
    pool_queries: bool = True,
) -> pd.DataFrame:
    """Concatenate long attention rows for all idx in mean_attn_by_idx."""
    frames = [
        attn_matrix_to_long_df(mat, idx, pool_queries=pool_queries)
        for idx, mat in sorted(mean_attn_by_idx.items())
    ]
    if not frames:
        return pd.DataFrame(columns=["idx", "freq_bin", "attn_score"])
    return pd.concat(frames, ignore_index=True)


def summarize_freq_bin_distribution(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per freq_bin summary of pooled attention scores."""
    if long_df.empty:
        return pd.DataFrame(
            columns=[
                "freq_bin",
                "n_points",
                "n_samples",
                "mean",
                "std",
                "q25",
                "median",
                "q75",
                "min",
                "max",
            ]
        )
    agg = (
        long_df.groupby("freq_bin", as_index=False)
        .agg(
            n_points=("attn_score", "size"),
            n_samples=("idx", "nunique"),
            mean=("attn_score", "mean"),
            std=("attn_score", "std"),
            q25=("attn_score", lambda s: float(s.quantile(0.25))),
            median=("attn_score", "median"),
            q75=("attn_score", lambda s: float(s.quantile(0.75))),
            min=("attn_score", "min"),
            max=("attn_score", "max"),
        )
        .sort_values("freq_bin")
    )
    return agg


def _dataset_display_label(dataset: str) -> str:
    if dataset == "e_coli":
        return r"$\it{E.\ coli}$"
    if dataset == "s_aureus":
        return r"$\it{S.\ aureus}$"
    return dataset


def _freq_bin_tick_label(freq_bin: int) -> str:
    """X-axis tick label for frequency bin (index only, no prefix)."""
    return str(int(freq_bin))


TITLE_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 12
VIOLIN_FILL = "#86CFC5"
BOX_FILL = "#86CFC5"


def plot_latent_query_freq_distribution(
    long_df: pd.DataFrame,
    out_png: str,
    *,
    kind: Literal["box", "violin"] = "box",
    dataset: str = "",
    n_seeds: int = 10,
    title: Optional[str] = None,
) -> None:
    """
    Box or violin plot: x=freq_bin, y=attn_score (queries pooled across test samples).
    """
    if long_df.empty:
        Logger.warning(f"Skipped {kind} plot (empty df): {out_png}")
        return

    work = long_df.copy()
    work["freq_bin"] = work["freq_bin"].astype(int)
    work["freq_label"] = work["freq_bin"].map(_freq_bin_tick_label)

    n_samples = int(work["idx"].nunique())
    ds_label = _dataset_display_label(dataset) if dataset else ""
    if title is None:
        title = (
            f"Latent-query attention by frequency bin — {ds_label} "
            f"(n={n_samples} test samples, {n_seeds} seeds mean, queries pooled)"
        )

    freq_order = sorted(work["freq_bin"].unique())
    freq_labels = [_freq_bin_tick_label(b) for b in freq_order]
    work["freq_label"] = pd.Categorical(work["freq_label"], categories=freq_labels, ordered=True)

    fig_w = max(10, min(24, len(freq_order) * 0.55))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    fill = VIOLIN_FILL if kind == "violin" else BOX_FILL

    if kind == "violin":
        sns.violinplot(
            data=work,
            x="freq_label",
            y="attn_score",
            order=freq_labels,
            inner="box",
            cut=0,
            linewidth=0.8,
            ax=ax,
            color=fill,
        )
    else:
        sns.boxplot(
            data=work,
            x="freq_label",
            y="attn_score",
            order=freq_labels,
            fliersize=1.5,
            linewidth=0.8,
            ax=ax,
            color=fill,
        )

    ax.set_xlabel("Frequency bin", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Cross-attention score", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, color="black")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    Logger.info(f"[saved] {out_png}")


def plot_latent_query_freq_distribution_combined(
    long_dfs: Dict[str, pd.DataFrame],
    out_png: str,
    *,
    kind: Literal["box", "violin"] = "box",
    n_seeds: int = 10,
    title: Optional[str] = None,
) -> None:
    """Side-by-side box/violin plots for multiple datasets."""
    datasets = [ds for ds, df in long_dfs.items() if not df.empty]
    if not datasets:
        Logger.warning(f"Skipped combined {kind} plot (no data): {out_png}")
        return

    if title is None:
        title = (
            f"Latent-query attention by frequency bin "
            f"({n_seeds} seeds mean, queries pooled)"
        )

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        work = long_dfs[dataset].copy()
        work["freq_bin"] = work["freq_bin"].astype(int)
        freq_order = sorted(work["freq_bin"].unique())
        freq_labels = [_freq_bin_tick_label(b) for b in freq_order]
        work["freq_label"] = work["freq_bin"].map(_freq_bin_tick_label)
        work["freq_label"] = pd.Categorical(work["freq_label"], categories=freq_labels, ordered=True)

        fill = VIOLIN_FILL if kind == "violin" else BOX_FILL
        if kind == "violin":
            sns.violinplot(
                data=work,
                x="freq_label",
                y="attn_score",
                order=freq_labels,
                inner="box",
                cut=0,
                linewidth=0.8,
                ax=ax,
                color=fill,
            )
        else:
            sns.boxplot(
                data=work,
                x="freq_label",
                y="attn_score",
                order=freq_labels,
                fliersize=1.5,
                linewidth=0.8,
                ax=ax,
                color=fill,
            )

        n_samples = int(work["idx"].nunique())
        ax.set_title(
            f"{_dataset_display_label(dataset)} (n={n_samples})",
            fontsize=TITLE_FONTSIZE,
            color="black",
        )
        ax.set_xlabel("Frequency bin", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Cross-attention score", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(title, y=1.02, fontsize=TITLE_FONTSIZE)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    Logger.info(f"[saved] {out_png}")
