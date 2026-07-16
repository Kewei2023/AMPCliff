#!/usr/bin/env python
# maintained by kewei li
# -*- coding: utf-8 -*-
"""
Split ablation metrics CSV into Excel file with multiple sheets.
"""

import pandas as pd
import os
from pathlib import Path


def _default_ablation_root(repo_root: Path) -> Path:
    env_root = os.environ.get("AMPCLIFF_ABLATION_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (repo_root / "outputs" / "ablation").resolve()


def _write_xlsx(df: pd.DataFrame, output_excel: Path) -> None:
    sheets = [
        ('s_aureus_esm2_t6', 's_aureus', 'esm2_t6'),
        ('e_coli_esm2_t6', 'e_coli', 'esm2_t6'),
        ('s_aureus_esm2_t12', 's_aureus', 'esm2_t12'),
        ('e_coli_esm2_t12', 'e_coli', 'esm2_t12'),
    ]
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        for sheet_name, dataset, model in sheets:
            sheet_df = df[(df['dataset'] == dataset) & (df['model'] == model)]
            sheet_df = sheet_df.sort_values(['experiment_type', 'config_name'])
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Sheet '{sheet_name}': {len(sheet_df)} experiments")


def main():
    """Main function."""
    repo_root = Path(__file__).resolve().parent
    ablation_root = _default_ablation_root(repo_root)
    input_csv = ablation_root / 'ablation_metrics_summary.csv'
    output_excel = ablation_root / 'ablation_metrics_by_task.xlsx'

    print("=" * 60)
    print("Splitting Ablation Metrics to Excel")
    print("=" * 60)

    df = pd.read_csv(input_csv)
    print(f"Total experiments: {len(df)}")

    try:
        _write_xlsx(df, output_excel)
        print(f"\nOutput saved to: {output_excel}")
    except PermissionError:
        alt = output_excel.with_name('ablation_metrics_by_task_new.xlsx')
        _write_xlsx(df, alt)
        print(
            f"\nPermission denied writing {output_excel}; "
            f"wrote {alt} instead (close Excel / fix mount permissions and re-run)."
        )
    print("=" * 60)


if __name__ == '__main__':
    main()
