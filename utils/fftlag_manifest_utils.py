# maintained by kewei li
"""Shared peptide manifest and checkpoint helpers for FFT-LAG mechanism experiments."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


def idx_from_name2id(name2id) -> int:
    """Extract sample Idx from collated name2id (seqName -> Idx list)."""
    for _name, ids in name2id.items():
        if isinstance(ids, (list, tuple)):
            return int(ids[0])
        return int(ids)
    raise ValueError("empty name2id in batch")


def idx_list_from_name2id(name2id) -> list:
    """All Idx values in a collated batch."""
    idxs = []
    for _name, ids in name2id.items():
        if isinstance(ids, (list, tuple)):
            idxs.extend(int(x) for x in ids)
        else:
            idxs.append(int(ids))
    return idxs


def batch_matches_manifest(name2id, allowed_idx: set) -> bool:
    """True if any sample in the batch is in the manifest."""
    return any(i in allowed_idx for i in idx_list_from_name2id(name2id))


def load_peptide_manifest(path: str) -> Tuple[Set[int], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    idx_list = {int(x) for x in payload["idx_list"]}
    return idx_list, payload


def default_manifest_dir() -> Path:
    return (
        REPO_ROOT
        / "outputs/ablation_new_data/_amp_knockout_seed_runs/_peptide_manifest"
    )


def manifest_path(
    model_version: str,
    dataset: str,
    diff: int = 5,
    out_dir: Optional[Path] = None,
) -> Path:
    base = out_dir or default_manifest_dir()
    return base / f"{model_version}_{dataset}_diff{diff}.json"


def ensure_peptide_manifest(
    model_version: str,
    dataset: str,
    *,
    diff: int = 5,
    threshold: float = 0.9,
    condition: str = "blosum62 average",
    n_peptides: int = 5,
    subset_seed: int = 42,
    out_dir: Optional[Path] = None,
    python_bin: str = sys.executable,
    dry_run: bool = False,
) -> Path:
    out = manifest_path(model_version, dataset, diff=diff, out_dir=out_dir)
    if out.is_file():
        return out
    if dry_run:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "evaluation_scripts/select_knockout_peptide_subset.py"
    cmd = [
        python_bin,
        str(script),
        "--model-version",
        model_version,
        "--dataset",
        dataset,
        "--diff",
        str(diff),
        "--threshold",
        str(threshold),
        "--condition",
        condition,
        "--n-peptides",
        str(n_peptides),
        "--subset-seed",
        str(subset_seed),
        "--out-dir",
        str(out.parent),
    ]
    subprocess.run(cmd, check=True)
    return out


def resolve_ckpt_dir(
    ablation_root: Path,
    model_version: str,
    pooling: str,
    dataset: str,
    seed: int,
    diff: int = 5,
) -> Optional[Path]:
    """Return model_step_* directory containing data/model.pth, or None."""
    exp_dir = ablation_root / f"{model_version}_{pooling}_{dataset}_diff{diff}"
    seed_dir = exp_dir / f"seed_{seed}"
    if not seed_dir.is_dir():
        return None
    matches = list(seed_dir.rglob("model.pth"))
    if not matches:
        return None
    return matches[0].parent.parent


def resolve_ablation_root(repo_root: Optional[Path] = None) -> Path:
    root = repo_root or REPO_ROOT
    for name in ("ablation_new_data", "ablation-new-data"):
        candidate = root / "outputs" / name
        if candidate.is_dir():
            return candidate
    return root / "outputs" / "ablation_new_data"


def get_analysis_filter(cfg) -> Tuple[Optional[Set[int]], Optional[int]]:
    """Read cfg.analysis for peptide manifest / max_samples filtering."""
    analysis = cfg.get("analysis", None)
    if analysis is None:
        return None, None

    allowed_idx: Optional[Set[int]] = None
    manifest_path_str = str(getattr(analysis, "peptide_manifest", "") or "").strip()
    if manifest_path_str:
        allowed_idx, _ = load_peptide_manifest(manifest_path_str)

    max_samples = getattr(analysis, "max_samples", None)
    if max_samples is not None:
        max_samples = int(max_samples)

    return allowed_idx, max_samples
