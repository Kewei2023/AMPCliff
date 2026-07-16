#!/usr/bin/env python3
# maintained by kewei li
"""
Path Helper Module
Unified path replacement for diff/condition/threshold/dataset placeholders
"""

from typing import Optional


def resolve_path(template_path: str,
                diff: int,
                condition: str,
                threshold: str,
                dataset: Optional[str] = None) -> str:
    """
    Resolve path by replacing placeholders with actual values

    Placeholders:
        {diff} → diff value (integer)
        {condition} → condition value (e.g., "blosum62 average")
        {threshold} → threshold value (e.g., "0.9")
        {dataset} → dataset value (e.g., "e_coli", "s_aureus")

    Args:
        template_path: Path template with placeholders
        diff: Difficulty difference value
        condition: Condition value
        threshold: Threshold value
        dataset: Dataset name (optional)

    Returns:
        Resolved path with all placeholders replaced
    """
    resolved = template_path

    # Replace placeholders in order
    resolved = resolved.replace("{diff}", str(diff))
    resolved = resolved.replace("{condition}", condition)
    resolved = resolved.replace("{threshold}", threshold)

    if dataset is not None:
        resolved = resolved.replace("{dataset}", dataset)

    return resolved


def get_data_paths(cfg: dict, task_type: str, diff: int,
                   condition: str, threshold: str, dataset: Optional[str] = None) -> dict:
    """
    Get all data file paths (train, valid, test, all_train) from config

    Args:
        cfg: Configuration dictionary (from OmegaConf)
        task_type: Task type (e.g., "regression", "classification")
        diff: Difficulty difference value
        condition: Condition value
        threshold: Threshold value
        dataset: Dataset name (optional)

    Returns:
        Dictionary with keys: 'all_train', 'train', 'valid', 'test'
    """
    data_config = cfg['data'][task_type]['fix']

    return {
        'all_train': resolve_path(data_config.get('all_train_file', ''), diff, condition, threshold, dataset),
        'train': resolve_path(data_config.get('train_file', ''), diff, condition, threshold, dataset),
        'valid': resolve_path(data_config.get('valid_file', ''), diff, condition, threshold, dataset),
        'test': resolve_path(data_config.get('test_file', ''), diff, condition, threshold, dataset),
    }
