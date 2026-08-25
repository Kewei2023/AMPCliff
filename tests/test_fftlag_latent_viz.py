"""Tests for training-aligned FFT-LAG latent visualization utilities."""

import numpy as np
import pytest
import torch

import torch.nn as nn

from AMPCliff.utils.fftlag_latent_viz import (
    aggregate_latent_out_cosine,
    attn_matrix_to_wide_df,
    attn_matrix_to_long_df,
    collect_sample_idxs,
    compute_attn_band_deviation,
    compute_gate_input_band_profiles,
    compute_gate_input_compare,
    compute_latent_out_contribution,
    compute_query_band_mass,
    compute_weighted_band_readout,
    expected_band_mass,
    export_attn_matrix_wide_csv,
    filter_by_idx,
    gate_input_from_freq_tokens,
    gate_input_from_latent_out,
    gate_input_mode_from_pooling,
    mean_latent_out_offdiag_cosine,
    plot_gate_input_latent_vs_freq,
    plot_raw_attn_heatmap,
    render_primary_plots_per_sample,
    sample_attn_matrix,
    summarize_latent_query_discriminability,
)


def test_expected_band_mass_sums_to_one():
    exp = expected_band_mass(num_freq=15, k_bands=8, band_mode="uniform")
    assert exp.shape == (8,)
    assert np.isclose(exp.sum(), 1.0)


def test_uniform_attention_deviation_near_zero():
    L, F = 4, 15
    w = torch.full((L, F), 1.0 / F)
    dev_df = compute_attn_band_deviation(w, k_bands=8, base=4, band_mode="uniform")
    assert np.allclose(dev_df["deviation"].values, 0.0, atol=1e-6)


def test_identical_latent_queries_zero_contribution():
    L, D = 4, 8
    v = torch.randn(D)
    lo = v.unsqueeze(0).expand(L, -1).clone()
    contrib = compute_latent_out_contribution(lo)
    assert np.allclose(contrib["l2_deviation"].values, 0.0, atol=1e-5)
    assert np.allclose(contrib["cos_to_gate_input"].values, 1.0, atol=1e-5)


def test_weighted_readout_scales_with_energy():
    L, F, D = 2, 8, 4
    w = torch.ones(L, F) / F
    ft_low = torch.zeros(F, D * 2)
    ft_low[0, :] = 10.0
    ft_high = torch.zeros(F, D * 2)
    ft_high[-1, :] = 10.0

    r_low = compute_weighted_band_readout(w, ft_low, k_bands=4, base=4, band_mode="uniform")
    r_high = compute_weighted_band_readout(w, ft_high, k_bands=4, base=4, band_mode="uniform")
    mass_low_b0 = r_low.loc[r_low["band"] == 0, "weighted_mass"].iloc[0]
    mass_high_b0 = r_high.loc[r_high["band"] == 0, "weighted_mass"].iloc[0]
    last_band = int(r_high["band"].max())
    mass_high_last = r_high.loc[r_high["band"] == last_band, "weighted_mass"].iloc[0]
    assert mass_low_b0 > mass_high_b0
    assert mass_high_last > mass_high_b0


def test_aggregate_cosine_identical_queries():
    lo = torch.randn(3, 16)
    lo = lo[0].unsqueeze(0).expand(3, -1).clone()
    mat = aggregate_latent_out_cosine([lo])
    assert mat.shape == (3, 3)
    assert np.allclose(mat, 1.0, atol=1e-5)


def test_sample_attn_matrix_batch_dim():
    w = torch.randn(1, 4, 10).softmax(dim=-1)
    out = sample_attn_matrix(w)
    assert out.shape == (4, 10)


def test_gate_input_mode_from_pooling_v3():
    assert gate_input_mode_from_pooling("fft_latent_attn_gate_v3") == "concat"
    assert gate_input_mode_from_pooling("fft_latent_attn_gate_v3_1") == "concat"
    assert gate_input_mode_from_pooling("fft_latent_attn_gate_v3_2") == "concat"
    assert gate_input_mode_from_pooling("fft_latent_attn_gate_v3_3") == "concat"
    assert gate_input_mode_from_pooling("fft_latent_attn_gate") == "mean"


def test_gate_input_concat_shape():
    lo = torch.randn(4, 8)
    gi = gate_input_from_latent_out(lo, mode="concat")
    assert gi.shape == (32,)


def test_mean_offdiag_cosine_identical_queries():
    lo = torch.randn(3, 16)
    lo = lo[0].unsqueeze(0).expand(3, -1).clone()
    mean_off, max_off = mean_latent_out_offdiag_cosine(lo)
    assert mean_off > 0.99
    assert max_off > 0.99


def test_summarize_discriminability_keys():
    lo = torch.randn(4, 8)
    w = torch.softmax(torch.randn(4, 12), dim=-1)
    out = summarize_latent_query_discriminability(
        lo, w, idx=1, k_bands=4, base=4, band_mode="uniform", mode="concat"
    )
    assert "mean_latent_out_offdiag_cosine" in out
    assert "mean_query_cosine_distance" in out


def test_gate_input_latent_matches_freq_when_vectors_are_identical():
    D = 8
    v = torch.randn(D)
    lo = v.unsqueeze(0).expand(4, -1).clone()
    ft = v.unsqueeze(0).expand(12, -1).clone()
    gi_l = gate_input_from_latent_out(lo)
    gi_f = gate_input_from_freq_tokens(ft)
    cos = float(torch.dot(gi_l, gi_f) / (gi_l.norm() * gi_f.norm() + 1e-8))
    assert cos > 0.99


def test_gate_input_band_profile_low_freq_dominant():
    F, D = 12, 4
    w = torch.ones(2, F) / F
    ft = torch.zeros(F, D * 2)
    ft[0, :] = 5.0
    prof = compute_gate_input_band_profiles(w, ft, k_bands=4, base=4, band_mode="uniform")
    b0 = prof.loc[prof["band"] == 0, "freq_uniform_pool_energy"].iloc[0]
    b3 = prof.loc[prof["band"] == 3, "freq_uniform_pool_energy"].iloc[0]
    assert b0 > b3


def test_compute_gate_input_compare_keys():
    D = 16
    lo = torch.randn(3, D)
    ft = torch.randn(10, D)
    freq_gate = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
    out = compute_gate_input_compare(lo, ft, freq_gate)
    assert "cos_latent_freq" in out
    assert "cos_gate_latent_vs_freq_cf" in out
    assert "gi_latent_l2" in out


def test_filter_by_idx_and_collect_sample_idxs():
    df = __import__("pandas").DataFrame({"idx": [1, 1, 2], "query": [0, 1, 0], "band": [0, 0, 0], "x": [1.0, 2.0, 3.0]})
    sub = filter_by_idx(df, 1)
    assert len(sub) == 2
    assert set(sub["idx"]) == {1}
    idxs = collect_sample_idxs(sub, sub, sub, latent_out_by_idx={1: torch.randn(2, 8)})
    assert idxs == [1]


def test_pivot_query_band_rejects_multiple_idx():
    import pandas as pd
    from AMPCliff.utils.fftlag_latent_viz import _pivot_query_band

    df = pd.DataFrame({
        "idx": [1, 2],
        "query": [0, 0],
        "band": [0, 0],
        "attention_mass": [0.1, 0.2],
    })
    with pytest.raises(ValueError, match="one peptide"):
        _pivot_query_band(df, "attention_mass")


def test_plot_gate_input_latent_vs_freq_single_peptide(tmp_path):
    import pandas as pd

    compare_df = pd.DataFrame([{
        "idx": 42,
        "cos_latent_freq": 0.05,
        "cos_gate_latent_vs_freq_cf": 0.98,
    }])
    out_png = tmp_path / "gate.png"
    plot_gate_input_latent_vs_freq(compare_df, str(out_png))
    assert out_png.is_file()


def _mini_exp4_tables(idx: int, k_bands: int = 4):
    import pandas as pd

    w = torch.ones(2, 8) / 8
    ft = torch.randn(8, 8)
    mass = compute_query_band_mass(w, k_bands=k_bands, base=4, band_mode="uniform")
    mass["idx"] = idx
    dev = compute_attn_band_deviation(w, k_bands=k_bands, base=4, band_mode="uniform")
    dev["idx"] = idx
    readout = compute_weighted_band_readout(w, ft, k_bands=k_bands, base=4, band_mode="uniform")
    readout["idx"] = idx
    lo = torch.randn(2, 8)
    contrib = compute_latent_out_contribution(lo)
    contrib["idx"] = idx
    freq_gate = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
    cmp_row = compute_gate_input_compare(lo, ft, freq_gate)
    cmp_row["idx"] = idx
    compare_df = pd.DataFrame([cmp_row])
    band = compute_gate_input_band_profiles(w, ft, k_bands=k_bands, base=4, band_mode="uniform")
    band["idx"] = idx
    return mass, dev, readout, contrib, compare_df, band, lo


def test_plot_raw_attn_heatmap(tmp_path):
    w = torch.softmax(torch.randn(4, 12), dim=-1)
    out_png = tmp_path / "attn_score_raw.png"
    out_csv = tmp_path / "attn_score_raw.csv"
    plot_raw_attn_heatmap(w, str(out_png), out_csv=str(out_csv))
    assert out_png.is_file()
    assert out_csv.is_file()


def test_attn_matrix_to_wide_df():
    w = torch.softmax(torch.randn(3, 5), dim=-1)
    df = attn_matrix_to_wide_df(w)
    assert list(df.columns) == ["query", "freq_0", "freq_1", "freq_2", "freq_3", "freq_4"]
    assert len(df) == 3
    assert list(df["query"]) == [0, 1, 2]
    np.testing.assert_allclose(df.iloc[0, 1:].to_numpy(), w[0].numpy(), rtol=1e-5)


def test_export_attn_matrix_wide_csv(tmp_path):
    w = torch.softmax(torch.randn(2, 4), dim=-1)
    out_csv = tmp_path / "attn_score_raw.csv"
    df = export_attn_matrix_wide_csv(w, str(out_csv))
    assert out_csv.is_file()
    loaded = __import__("pandas").read_csv(out_csv)
    assert list(loaded.columns) == list(df.columns)
    np.testing.assert_allclose(loaded.to_numpy(), df.to_numpy(), rtol=1e-5)


def test_render_primary_plots_per_sample_raw_attn_score(tmp_path):
    tables_a = _mini_exp4_tables(10)
    mass, dev, readout, contrib, compare, band, lo = tables_a
    w = torch.softmax(torch.randn(2, 8), dim=-1)
    attn_weights_by_idx = {10: w}

    render_primary_plots_per_sample(
        contrib_df=contrib,
        dev_df=dev,
        readout_df=readout,
        latent_out_by_idx={10: lo},
        dataset_label="test_ds",
        out_base=str(tmp_path),
        primary_plots=("raw_attn_score",),
        attn_weights_by_idx=attn_weights_by_idx,
    )

    sub = tmp_path / "per_sample" / "idx_10"
    assert (sub / "attn_score_raw.png").is_file()
    assert (sub / "attn_score_raw.csv").is_file()


def test_render_primary_plots_per_sample_two_peptides(tmp_path):
    tables_a = _mini_exp4_tables(10)
    tables_b = _mini_exp4_tables(20)
    mass = __import__("pandas").concat([tables_a[0], tables_b[0]], ignore_index=True)
    dev = __import__("pandas").concat([tables_a[1], tables_b[1]], ignore_index=True)
    readout = __import__("pandas").concat([tables_a[2], tables_b[2]], ignore_index=True)
    contrib = __import__("pandas").concat([tables_a[3], tables_b[3]], ignore_index=True)
    compare = __import__("pandas").concat([tables_a[4], tables_b[4]], ignore_index=True)
    band = __import__("pandas").concat([tables_a[5], tables_b[5]], ignore_index=True)
    latent_out_by_idx = {10: tables_a[6], 20: tables_b[6]}

    render_primary_plots_per_sample(
        contrib_df=contrib,
        dev_df=dev,
        readout_df=readout,
        latent_out_by_idx=latent_out_by_idx,
        dataset_label="test_ds",
        out_base=str(tmp_path),
        primary_plots=("query_contribution", "latent_out_cosine"),
    )

    for idx in (10, 20):
        sub = tmp_path / "per_sample" / f"idx_{idx}"
        assert (sub / "latent_query_contribution.png").is_file()
        assert (sub / "latent_out_cosine_heatmap.png").is_file()


def test_attn_matrix_to_long_df_pooled_rows():
    w = torch.softmax(torch.randn(8, 15), dim=-1)
    df = attn_matrix_to_long_df(w, idx=42, pool_queries=True)
    assert len(df) == 8 * 15
    assert set(df.columns) == {"idx", "freq_bin", "attn_score"}
    assert df["idx"].nunique() == 1
    assert df["freq_bin"].nunique() == 15


def test_build_pooled_freq_distribution_and_summary():
    from AMPCliff.utils.fftlag_latent_viz import (
        build_pooled_freq_distribution_df,
        mean_attn_matrices,
        summarize_freq_bin_distribution,
    )

    mats = [torch.softmax(torch.randn(8, 15), dim=-1) for _ in range(2)]
    mean_by_idx = {
        1: mean_attn_matrices(mats),
        2: mean_attn_matrices(mats),
        3: mean_attn_matrices(mats),
    }
    long_df = build_pooled_freq_distribution_df(mean_by_idx, pool_queries=True)
    assert len(long_df) == 3 * 8 * 15
    summary = summarize_freq_bin_distribution(long_df)
    assert len(summary) == 15
    assert summary.loc[0, "n_points"] == 3 * 8


def test_aggregate_attn_across_seeds_from_pt(tmp_path):
    from AMPCliff.utils.fftlag_latent_viz import aggregate_attn_by_idx_across_seeds

    analysis_root = tmp_path / "analysis"
    for seed in (0, 1):
        exp_dir = analysis_root / f"seed_{seed}" / "e_coli" / "exp4_latent_fulltest"
        exp_dir.mkdir(parents=True)
        aw = torch.softmax(torch.randn(8, 15), dim=-1)
        payload = {"samples": [{"idx": 10, "attn_weights": aw}, {"idx": 20, "attn_weights": aw}]}
        torch.save(payload, exp_dir / "latent_attn_weights.pt")

    seed_dirs = [analysis_root / "seed_0", analysis_root / "seed_1"]
    mean_by_idx, warnings = aggregate_attn_by_idx_across_seeds(
        seed_dirs, "e_coli", subdir="exp4_latent_fulltest", min_seeds=2
    )
    assert not warnings
    assert set(mean_by_idx.keys()) == {10, 20}
    assert mean_by_idx[10].shape == (8, 15)


def test_plot_latent_query_freq_distribution_smoke(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from AMPCliff.utils.fftlag_latent_viz import (
        build_pooled_freq_distribution_df,
        plot_latent_query_freq_distribution,
        plot_latent_query_freq_distribution_combined,
    )

    mean_by_idx = {
        i: torch.softmax(torch.randn(8, 15), dim=-1) for i in range(3)
    }
    long_df = build_pooled_freq_distribution_df(mean_by_idx, pool_queries=True)

    box_png = tmp_path / "box.png"
    violin_png = tmp_path / "violin.png"
    plot_latent_query_freq_distribution(long_df, str(box_png), kind="box", dataset="e_coli")
    plot_latent_query_freq_distribution(long_df, str(violin_png), kind="violin", dataset="e_coli")
    assert box_png.is_file()
    assert violin_png.is_file()

    combined_png = tmp_path / "combined.png"
    plot_latent_query_freq_distribution_combined(
        {"e_coli": long_df, "s_aureus": long_df},
        str(combined_png),
        kind="box",
    )
    assert combined_png.is_file()
