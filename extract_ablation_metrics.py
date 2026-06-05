#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract metrics from ablation experiments and save to CSV.

This script extracts train/valid/test metrics (pearson, spearman, recall)
from ablation runs under outputs/ablation/.

Usage:
  # All experiment groups under outputs/ablation/
  python extract_ablation_metrics.py

  # Single group (e.g. spectral_anchor + DC ablation)
  python extract_ablation_metrics.py \\
    --group-dir outputs/ablation/esm2_t6_spectral_anchor_dc_ablation_s_aureus_diff5 \\
    --output outputs/ablation/esm2_t6_spectral_anchor_dc_ablation_s_aureus_diff5/metrics_summary.csv
"""

import argparse
import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from scipy.stats import spearmanr, pearsonr

# Match downstream_train.py: Metrics(..., topK=50) for recall in logs
DEFAULT_RECALL_TOPK = 50


def _default_ablation_root(repo_root: Path) -> Path:
    env_root = os.environ.get("AMPCLIFF_ABLATION_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (repo_root / "outputs" / "ablation").resolve()


def _cal_recall(y_pred: np.ndarray, y_true: np.ndarray, top: int) -> int:
    """Same logic as AMPCliff.utils.metrics.cal_recall (top-k index intersection)."""
    a_sort_idx = y_pred.argsort()
    b_sort_idx = y_true.argsort()
    top = min(top, len(y_pred), len(y_true))
    if top <= 0:
        return 0
    return len(set(b_sort_idx[-top:].tolist()).intersection(a_sort_idx[-top:].tolist()))


def parse_log_file(log_path: Path) -> Dict[str, float]:
    """Parse downstream_train.log to extract best metrics.
    
    Returns metrics from the best checkpoint (based on valid spearman).
    """
    metrics = {
        'train_pearson': None,
        'train_spearman': None,
        'train_recall': None,
        'valid_pearson': None,
        'valid_spearman': None,
        'valid_recall': None,
    }
    
    if not log_path.exists():
        return metrics
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Track best valid spearman and corresponding train metrics
    best_valid_spearman = -np.inf
    best_step = None
    
    # Store all metrics by step
    step_metrics = {}
    
    for line in lines:
        # Parse train metrics
        train_match = re.search(
            r'train \| step: (\d+) \| train-pearson: ([\d.]+)', line
        )
        if train_match:
            step = int(train_match.group(1))
            if step not in step_metrics:
                step_metrics[step] = {}
            step_metrics[step]['train_pearson'] = float(train_match.group(2))
        
        train_spearman_match = re.search(
            r'train \| step: \d+ \| train-spearman: ([\d.]+)', line
        )
        if train_spearman_match:
            step = int(re.search(r'train \| step: (\d+)', line).group(1))
            if step not in step_metrics:
                step_metrics[step] = {}
            step_metrics[step]['train_spearman'] = float(train_spearman_match.group(1))
        
        train_recall_match = re.search(
            r'train \| step: \d+ \| train-recall: ([\d.]+)', line
        )
        if train_recall_match:
            step = int(re.search(r'train \| step: (\d+)', line).group(1))
            if step not in step_metrics:
                step_metrics[step] = {}
            step_metrics[step]['train_recall'] = float(train_recall_match.group(1))
        
        # Parse valid metrics
        valid_match = re.search(
            r'valid \| step: (\d+) \| valid-pearson: ([\d.]+)', line
        )
        if valid_match:
            step = int(valid_match.group(1))
            if step not in step_metrics:
                step_metrics[step] = {}
            step_metrics[step]['valid_pearson'] = float(valid_match.group(2))
        
        valid_spearman_match = re.search(
            r'valid \| step: \d+ \| valid-spearman: ([\d.]+)', line
        )
        if valid_spearman_match:
            step = int(re.search(r'valid \| step: (\d+)', line).group(1))
            if step not in step_metrics:
                step_metrics[step] = {}
            step_metrics[step]['valid_spearman'] = float(valid_spearman_match.group(1))
        
        valid_recall_match = re.search(
            r'valid \| step: \d+ \| valid-recall: ([\d.]+)', line
        )
        if valid_recall_match:
            step = int(re.search(r'valid \| step: (\d+)', line).group(1))
            if step not in step_metrics:
                step_metrics[step] = {}
            step_metrics[step]['valid_recall'] = float(valid_recall_match.group(1))
    
    # Find step with best valid spearman
    for step, step_data in step_metrics.items():
        if 'valid_spearman' in step_data:
            if step_data['valid_spearman'] > best_valid_spearman:
                best_valid_spearman = step_data['valid_spearman']
                best_step = step
    
    # Extract metrics from best step
    if best_step is not None and best_step in step_metrics:
        for key in ['train_pearson', 'train_spearman', 'train_recall',
                    'valid_pearson', 'valid_spearman', 'valid_recall']:
            if key in step_metrics[best_step]:
                metrics[key] = step_metrics[best_step][key]
    
    return metrics


def calculate_metrics_from_csv(
    csv_path: Path,
    pred_col: str = 'esm2_t6-pred',
    true_col: str = 'true',
    recall_topk: int = DEFAULT_RECALL_TOPK,
) -> Optional[Dict[str, float]]:
    """Calculate pearson, spearman, recall@topk from result CSV file."""
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        if pred_col not in df.columns or true_col not in df.columns:
            # Try to find pred column
            pred_cols = [c for c in df.columns if '-pred' in c]
            if pred_cols:
                pred_col = pred_cols[0]
            else:
                return None
        
        # Remove NaN values
        valid_mask = ~(df[pred_col].isna() | df[true_col].isna())
        valid_df = df[valid_mask]
        
        if len(valid_df) < 2:
            return None
        
        pred = valid_df[pred_col].values
        true = valid_df[true_col].values
        
        pearson, _ = pearsonr(pred, true)
        spearman, _ = spearmanr(pred, true)
        recall = float(_cal_recall(pred, true, recall_topk))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))

        return {
            'pearson': pearson,
            'spearman': spearman,
            'recall': recall,
            'recall_topk': float(recall_topk),
            'rmse': rmse,
        }
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None


# Hydra dir template from run_spectral_anchor_dc_ablation.sh:
# apply_${apply}_k${num_anchor}_fft${use_fft}_s${scale_dc.scale}_d${distill_vc.scale}
_SPECTRAL_ANCHOR_DC_RUN_RE = re.compile(
    r'^apply_(?P<dc_apply>scale_dc|distill_vc)_k(?P<num_anchor>\d+)'
    r'_fft(?P<use_fft>True|False)_s(?P<path_s>[\d.]+)_d(?P<path_d>[\d.]+)$'
)

_SA_V2_MULTI_HEAD_RE = re.compile(
    r'^h(?P<num_heads>\d+)_fft(?P<use_fft>true|false)_gated(?P<gated>true|false)$',
    re.IGNORECASE,
)
_SA_V2_CLASSIC_RE = re.compile(
    r'^k(?P<k>\d+)_fft(?P<use_fft>true|false)$',
    re.IGNORECASE,
)
_SA_V2_COMBINE_RE = re.compile(
    r'^k(?P<k>\d+)_(?P<combine>max|mean|soft)_fft(?P<use_fft>true|false)$',
    re.IGNORECASE,
)

# run_local_spectral_anchor_ablation.sh — core grid + toggle slugs
_LOCAL_SPECTRAL_GRID_RE = re.compile(
    r'^h(?P<num_heads>\d+)_k(?P<num_anchor>\d+)_adim(?P<analysis_dim>\d+)'
    r'_nfft(?P<stft_n_fft>\d+)_hop(?P<stft_hop_length>\d+)$'
)
_LOCAL_SPECTRAL_TOGGLE_RE = re.compile(
    r'^toggle_h(?P<base_h>\d+)_nfft(?P<base_nfft>\d+)'
    r'_c(?P<stft_center>true|false)_p(?P<use_phase>true|false)_g(?P<gated>true|false)$',
    re.IGNORECASE,
)


def iter_leaf_run_dirs(exp_group_dir: Path) -> List[tuple]:
    """Each run directory that contains downstream_train.log (posix rel path, Path)."""
    found: List[tuple] = []
    if not exp_group_dir.is_dir():
        return found
    for log_path in sorted(exp_group_dir.rglob('downstream_train.log')):
        if 'multirun' in log_path.parts:
            continue
        run_dir = log_path.parent
        try:
            rel = run_dir.relative_to(exp_group_dir).as_posix()
        except ValueError:
            continue
        if not rel or rel.startswith('.'):
            continue
        found.append((rel, run_dir))
    return found


def parse_experiment_config(rel_posix: str, parent_dir_name: str) -> Dict[str, str]:
    """Parse experiment configuration from run path relative to group dir (may contain '/')."""
    leaf = Path(rel_posix).name
    config: Dict[str, str] = {
        'experiment_type': '',
        'model': '',
        'dataset': '',
        'config_name': rel_posix,
    }

    parts = parent_dir_name.split('_')

    if parts[0] == 'esm2':
        config['model'] = f"{parts[0]}_{parts[1]}"
    else:
        config['model'] = parts[0]

    diff_m = re.search(r'diff(\d+)', parent_dir_name)
    if diff_m:
        config['diff'] = diff_m.group(1)

    if 's_aureus' in parent_dir_name:
        config['dataset'] = 's_aureus'
    elif 'e_coli' in parent_dir_name:
        config['dataset'] = 'e_coli'

    # Local spectral anchor (run_local_spectral_anchor_ablation*.sh): local_spectral_anchor/<slug>
    if 'local_spectral_anchor_ablation' in parent_dir_name:
        config['experiment_type'] = 'local_spectral_anchor_ablation'
        config['pooling'] = 'local_spectral_anchor'
        segs = rel_posix.split('/')
        if len(segs) == 2 and segs[0] == 'local_spectral_anchor':
            slug = segs[1]
            gm = _LOCAL_SPECTRAL_GRID_RE.match(slug)
            if gm:
                config['num_heads'] = gm.group('num_heads')
                config['num_anchor'] = gm.group('num_anchor')
                config['analysis_dim'] = gm.group('analysis_dim')
                config['stft_n_fft'] = gm.group('stft_n_fft')
                config['stft_hop_length'] = gm.group('stft_hop_length')
                config['run_slug_type'] = 'grid'
            else:
                tm = _LOCAL_SPECTRAL_TOGGLE_RE.match(slug)
                if tm:
                    config['num_heads'] = tm.group('base_h')
                    config['stft_n_fft'] = tm.group('base_nfft')
                    config['stft_center'] = (
                        'True' if tm.group('stft_center').lower() == 'true' else 'False'
                    )
                    config['use_phase'] = (
                        'True' if tm.group('use_phase').lower() == 'true' else 'False'
                    )
                    config['gated'] = (
                        'True' if tm.group('gated').lower() == 'true' else 'False'
                    )
                    config['run_slug_type'] = 'toggle'
        return config

    # Nested layout: e.g. multi_head_spectral/h8_ffttrue_gatedfalse
    if 'spectral_anchor_v2_ablation' in parent_dir_name:
        config['experiment_type'] = 'spectral_anchor_v2_ablation'
        segs = rel_posix.split('/')
        if len(segs) == 1 and segs[0] in ('attn', 'max', 'mean'):
            config['pooling'] = segs[0]
        elif len(segs) == 2:
            arm, run_leaf = segs[0], segs[1]
            if arm == 'spectral_anchor':
                m = _SA_V2_CLASSIC_RE.match(run_leaf)
                if m:
                    config['pooling'] = 'spectral_anchor'
                    config['k_value'] = m.group('k')
                    config['use_fft'] = (
                        'True' if m.group('use_fft').lower() == 'true' else 'False'
                    )
            elif arm == 'multi_head_spectral':
                m = _SA_V2_MULTI_HEAD_RE.match(run_leaf)
                if m:
                    config['pooling'] = 'multi_head_spectral'
                    config['num_heads'] = m.group('num_heads')
                    config['k_value'] = m.group('num_heads')
                    config['use_fft'] = (
                        'True' if m.group('use_fft').lower() == 'true' else 'False'
                    )
                    config['gated'] = (
                        'True' if m.group('gated').lower() == 'true' else 'False'
                    )
            elif arm == 'spectral_anchor_v2':
                m = _SA_V2_COMBINE_RE.match(run_leaf)
                if m:
                    config['pooling'] = 'spectral_anchor_v2'
                    config['k_value'] = m.group('k')
                    config['combine_mode'] = m.group('combine')
                    config['use_fft'] = (
                        'True' if m.group('use_fft').lower() == 'true' else 'False'
                    )
        return config

    if 'orthogonal' in parent_dir_name:
        config['experiment_type'] = 'orthogonal_constraint'
    elif 'pooling' in parent_dir_name:
        config['experiment_type'] = 'pooling_baseline'
    elif 'spectral_anchor_dc' in parent_dir_name:
        config['experiment_type'] = 'spectral_anchor_dc'

    sa_dc_m = _SPECTRAL_ANCHOR_DC_RUN_RE.match(leaf)
    if sa_dc_m:
        config['experiment_type'] = 'spectral_anchor_dc'
        config['dc_apply'] = sa_dc_m.group('dc_apply')
        config['num_anchor'] = sa_dc_m.group('num_anchor')
        config['use_fft'] = sa_dc_m.group('use_fft')
        config['path_s'] = sa_dc_m.group('path_s')
        config['path_d'] = sa_dc_m.group('path_d')

    if config['experiment_type'] == 'orthogonal_constraint':
        match = re.search(r'enabled_(True|False)_w([\d.]+)_t(\w+)_l(\[.*?\]|None)', leaf)
        if match:
            config['orthogonal_enabled'] = match.group(1)
            config['orthogonal_weight'] = match.group(2)
            config['orthogonal_type'] = match.group(3)
            config['orthogonal_layers'] = match.group(4)

    elif config['experiment_type'] == 'pooling_baseline':
        match = re.search(r'(\w+)_k(\d+)_fft(True|False)', leaf)
        if match:
            config['pooling'] = match.group(1)
            config['k_value'] = match.group(2)
            config['use_fft'] = match.group(3)

    return config


def extract_metrics_for_group(exp_group_dir: Path) -> List[dict]:
    """Extract metrics for one experiment group directory (contains run subfolders)."""
    results: List[dict] = []
    print(f"\nProcessing experiment group: {exp_group_dir.name}")

    for rel_path, exp_dir in iter_leaf_run_dirs(exp_group_dir):
        print(f"  Processing: {rel_path}")

        config = parse_experiment_config(rel_path, exp_group_dir.name)
        log_file = exp_dir / 'downstream_train.log'
        log_metrics = parse_log_file(log_file)

        test_csvs = list(exp_dir.glob('*-test_result.csv'))
        test_metrics: dict = {}
        if test_csvs:
            test_result = calculate_metrics_from_csv(test_csvs[0])
            if test_result:
                test_metrics['test_pearson'] = test_result['pearson']
                test_metrics['test_spearman'] = test_result['spearman']
                test_metrics['test_recall'] = test_result['recall']
                test_metrics['test_recall_topk'] = test_result['recall_topk']

        results.append({
            'experiment_group': exp_group_dir.name,
            'experiment_dir': rel_path,
            **config,
            **log_metrics,
            **test_metrics,
        })

    return results


def extract_all_ablation_metrics(ablation_root: Path) -> pd.DataFrame:
    """Extract metrics from all ablation experiments."""
    results: List[dict] = []

    if not ablation_root.exists():
        print(f"Ablation directory not found: {ablation_root}")
        return pd.DataFrame()

    for exp_group_dir in sorted(ablation_root.iterdir()):
        if not exp_group_dir.is_dir():
            continue
        results.extend(extract_metrics_for_group(exp_group_dir))

    return pd.DataFrame(results)


def extract_single_group_metrics(group_dir: Path) -> pd.DataFrame:
    """Extract metrics for a single experiment group path."""
    if not group_dir.is_dir():
        print(f"Group directory not found or not a directory: {group_dir}")
        return pd.DataFrame()
    return pd.DataFrame(extract_metrics_for_group(group_dir))


def main():
    """Main function to extract and save ablation metrics."""
    repo_root = Path(__file__).resolve().parent
    default_ablation_root = _default_ablation_root(repo_root)

    parser = argparse.ArgumentParser(
        description='Extract train/valid/test pearson, spearman, recall from ablation outputs.',
    )
    parser.add_argument(
        '--group-dir',
        type=Path,
        default=None,
        help='If set, only process this experiment group directory (contains apply_* run folders).',
    )
    parser.add_argument(
        '--ablation-root',
        type=Path,
        default=default_ablation_root,
        help='Root containing experiment group folders (ignored if --group-dir is set).',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output CSV path. Default: <group-dir>/metrics_summary.csv or ablation_metrics_summary.csv.',
    )
    args = parser.parse_args()

    if args.group_dir is not None:
        group_path = args.group_dir.expanduser().resolve()
        output_csv = (
            args.output.expanduser().resolve()
            if args.output is not None
            else group_path / 'metrics_summary.csv'
        )
        print("=" * 60)
        print("Extracting Ablation Experiment Metrics (single group)")
        print("=" * 60)
        df = extract_single_group_metrics(group_path)
    else:
        ablation_root = args.ablation_root.expanduser().resolve()
        output_csv = (
            args.output.expanduser().resolve()
            if args.output is not None
            else ablation_root / 'ablation_metrics_summary.csv'
        )
        print("=" * 60)
        print("Extracting Ablation Experiment Metrics")
        print("=" * 60)
        df = extract_all_ablation_metrics(ablation_root)
    
    if df.empty:
        print("\nNo experiments found!")
        return

    # Reorder columns for better readability
    column_order = [
        'experiment_group',
        'experiment_dir',
        'experiment_type',
        'model',
        'dataset',
        'diff',
        'config_name',
        # Pooling specific
        'pooling',
        'k_value',
        'use_fft',
        # Spectral anchor v2 ablation (multi-head / combine)
        'combine_mode',
        'gated',
        'num_heads',
        # Local spectral anchor (run_local_spectral_anchor_ablation*.sh)
        'analysis_dim',
        'stft_n_fft',
        'stft_hop_length',
        'stft_center',
        'use_phase',
        'run_slug_type',
        # Spectral anchor + DC (run_spectral_anchor_dc_ablation.sh)
        'dc_apply',
        'num_anchor',
        'path_s',
        'path_d',
        # Orthogonal specific
        'orthogonal_enabled',
        'orthogonal_weight',
        'orthogonal_type',
        'orthogonal_layers',
        # Metrics
        'train_pearson',
        'train_spearman',
        'train_recall',
        'valid_pearson',
        'valid_spearman',
        'valid_recall',
        'test_pearson',
        'test_spearman',
        'test_recall',
        'test_recall_topk',
    ]
    
    # Only include columns that exist
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    # Sort by experiment group and directory
    df = df.sort_values(['experiment_group', 'experiment_dir'])
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total experiments: {len(df)}")
    print(f"Output saved to: {output_csv}")
    print(f"\nColumn statistics:")
    print(f"  - Experiments with train metrics: {df['train_pearson'].notna().sum()}")
    print(f"  - Experiments with valid metrics: {df['valid_pearson'].notna().sum()}")
    print(f"  - Experiments with test metrics: {df['test_pearson'].notna().sum()}")
    
    print("\n" + "="*60)
    print("Preview")
    print("="*60)
    print(df.head(10).to_string())


if __name__ == '__main__':
    main()
