"""Tests for cross-seed Exp4 attn_score_raw aggregation."""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPTS = REPO_ROOT / "evaluation_scripts"
if str(EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPTS))

from aggregate_fftlag_exp4_attn_score_raw import aggregate_dataset, run_all


def _agg_png(analysis_root: Path, dataset: str, idx: int) -> Path:
    return (
        analysis_root
        / "aggregated"
        / dataset
        / "exp4"
        / "per_sample"
        / f"idx_{idx}"
        / "plots"
        / "mean_style"
        / "attn_score_raw.png"
    )


def _agg_csv(analysis_root: Path, dataset: str, idx: int) -> Path:
    return _agg_png(analysis_root, dataset, idx).with_suffix(".csv")


def _write_pt(exp4_dir: Path, idx: int, shape) -> None:
    exp4_dir.mkdir(parents=True, exist_ok=True)
    aw = torch.softmax(torch.randn(*shape), dim=-1)
    torch.save({"samples": [{"idx": idx, "attn_weights": aw}]}, exp4_dir / "latent_attn_weights.pt")


def test_aggregate_separate_datasets(tmp_path):
    analysis_root = tmp_path / "analysis"

    for ds, idx in (("e_coli", 234), ("s_aureus", 1681)):
        for seed in (0, 1):
            exp4 = analysis_root / f"seed_{seed}" / ds / "exp4_latent"
            _write_pt(exp4, idx=idx, shape=(3, 12))

    run, skip, fail, missing, shape_mismatch = run_all(
        analysis_root,
        datasets=["e_coli", "s_aureus"],
        seeds=["0", "1"],
        min_seeds=2,
    )

    assert run == 2
    assert skip == 0
    assert fail == 0
    assert missing == 0
    assert shape_mismatch == 0

    ec_png = _agg_png(analysis_root, "e_coli", 234)
    sa_png = _agg_png(analysis_root, "s_aureus", 1681)
    assert ec_png.is_file()
    assert sa_png.is_file()
    assert _agg_csv(analysis_root, "e_coli", 234).is_file()
    assert _agg_csv(analysis_root, "s_aureus", 1681).is_file()
    assert not _agg_png(analysis_root, "e_coli", 1681).parent.exists()
    assert not _agg_png(analysis_root, "s_aureus", 234).parent.exists()


def test_shape_mismatch_skipped(tmp_path):
    analysis_root = tmp_path / "analysis"
    _write_pt(analysis_root / "seed_0/e_coli/exp4_latent", idx=10, shape=(3, 8))
    _write_pt(analysis_root / "seed_1/e_coli/exp4_latent", idx=10, shape=(4, 8))

    run, skip, fail, missing, shape_mismatch = aggregate_dataset(
        analysis_root,
        "e_coli",
        [analysis_root / "seed_0", analysis_root / "seed_1"],
        min_seeds=2,
    )

    assert run == 0
    assert skip == 0
    assert fail == 0
    assert shape_mismatch == 1
