"""Tests for offline Exp4 attn_score_raw replot script."""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPTS = REPO_ROOT / "evaluation_scripts"
if str(EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPTS))

from plot_fftlag_exp4_attn_score_raw import run_all


def test_run_all_from_fake_pt(tmp_path):
    analysis_root = tmp_path / "analysis"
    exp4_dir = analysis_root / "seed_0" / "s_aureus" / "exp4_latent"
    exp4_dir.mkdir(parents=True)

    aw = torch.softmax(torch.randn(3, 10), dim=-1)
    payload = {
        "samples": [
            {"idx": 42, "attn_weights": aw, "latent_out": None, "latent_summary": None},
            {"idx": 99, "attn_weights": aw, "latent_out": None, "latent_summary": None},
        ]
    }
    torch.save(payload, exp4_dir / "latent_attn_weights.pt")

    run, skip, fail, missing = run_all(
        analysis_root,
        datasets=["s_aureus"],
        seeds=["0"],
        force=False,
        dry_run=False,
    )

    assert run == 2
    assert skip == 0
    assert fail == 0
    assert missing == 0
    assert (exp4_dir / "per_sample" / "idx_42" / "attn_score_raw.png").is_file()
    assert (exp4_dir / "per_sample" / "idx_42" / "attn_score_raw.csv").is_file()
    assert (exp4_dir / "per_sample" / "idx_99" / "attn_score_raw.png").is_file()
    assert (exp4_dir / "per_sample" / "idx_99" / "attn_score_raw.csv").is_file()


def test_run_all_backfills_missing_csv(tmp_path):
    analysis_root = tmp_path / "analysis"
    exp4_dir = analysis_root / "seed_1" / "e_coli" / "exp4_latent"
    out_dir = exp4_dir / "per_sample" / "idx_7"
    out_dir.mkdir(parents=True)
    (out_dir / "attn_score_raw.png").write_bytes(b"fake")

    aw = torch.softmax(torch.randn(2, 8), dim=-1)
    torch.save(
        {"samples": [{"idx": 7, "attn_weights": aw}]},
        exp4_dir / "latent_attn_weights.pt",
    )

    run, skip, fail, missing = run_all(
        analysis_root,
        datasets=["e_coli"],
        seeds=["1"],
        force=False,
        dry_run=False,
    )

    assert run == 1
    assert skip == 0
    assert fail == 0
    assert missing == 0
    assert (out_dir / "attn_score_raw.csv").is_file()


def test_run_all_skips_existing_png_and_csv(tmp_path):
    analysis_root = tmp_path / "analysis"
    exp4_dir = analysis_root / "seed_1" / "e_coli" / "exp4_latent"
    out_dir = exp4_dir / "per_sample" / "idx_7"
    out_dir.mkdir(parents=True)
    (out_dir / "attn_score_raw.png").write_bytes(b"fake")
    (out_dir / "attn_score_raw.csv").write_text("query,freq_0\n0,0.1\n")

    aw = torch.softmax(torch.randn(2, 8), dim=-1)
    torch.save(
        {"samples": [{"idx": 7, "attn_weights": aw}]},
        exp4_dir / "latent_attn_weights.pt",
    )

    run, skip, fail, missing = run_all(
        analysis_root,
        datasets=["e_coli"],
        seeds=["1"],
        force=False,
        dry_run=False,
    )

    assert run == 0
    assert skip == 1
    assert fail == 0
    assert missing == 0
