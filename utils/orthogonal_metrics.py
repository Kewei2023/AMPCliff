"""
Orthogonal Metrics Module for AMPCliff.

This module provides utilities for computing and logging orthogonality metrics
between layer representations.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union


def compute_mean_orthogonality(
    layer_representations: Union[torch.Tensor, List[torch.Tensor]],
    normalize: bool = True
) -> float:
    """
    Compute mean pairwise orthogonality between all layer pairs.

    Args:
        layer_representations: Layer outputs [B, D, L] or List[Tensor]
        normalize: Whether to L2 normalize before computing

    Returns:
        Mean absolute cosine similarity (0 = orthogonal, 1 = parallel)
    """
    from .orthogonal_constraint import OrthogonalConstraint

    constraint = OrthogonalConstraint(normalize=normalize)
    ortho_matrix = constraint.compute_orthogonality_matrix(layer_representations)

    # Get upper triangle (excluding diagonal)
    L = ortho_matrix.shape[0]
    mask = torch.triu(torch.ones(L, L, device=ortho_matrix.device), diagonal=1).bool()
    upper_tri = ortho_matrix[mask]

    # Return mean absolute similarity
    return float(upper_tri.abs().mean().item())


def compute_layerwise_orthogonality(
    layer_representations: Union[torch.Tensor, List[torch.Tensor]],
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compute orthogonality between adjacent layer pairs.

    Args:
        layer_representations: Layer outputs [B, D, L] or List[Tensor]
        normalize: Whether to L2 normalize before computing

    Returns:
        Dictionary mapping layer pair names to orthogonality scores
    """
    from .orthogonal_constraint import OrthogonalConstraint

    constraint = OrthogonalConstraint(normalize=normalize)
    ortho_matrix = constraint.compute_orthogonality_matrix(layer_representations)

    L = ortho_matrix.shape[0]
    layerwise_ortho = {}

    for i in range(L - 1):
        cos_sim = ortho_matrix[i, i + 1].item()
        layerwise_ortho[f"layer_{i}_vs_{i+1}"] = cos_sim

    return layerwise_ortho


def compute_orthogonality_statistics(
    layer_representations: Union[torch.Tensor, List[torch.Tensor]],
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compute comprehensive orthogonality statistics.

    Args:
        layer_representations: Layer outputs [B, D, L] or List[Tensor]
        normalize: Whether to L2 normalize before computing

    Returns:
        Dictionary with mean, max, min, std of orthogonality scores
    """
    from .orthogonal_constraint import OrthogonalConstraint

    constraint = OrthogonalConstraint(normalize=normalize)
    ortho_matrix = constraint.compute_orthogonality_matrix(layer_representations)

    # Get upper triangle (excluding diagonal)
    L = ortho_matrix.shape[0]
    mask = torch.triu(torch.ones(L, L, device=ortho_matrix.device), diagonal=1).bool()
    upper_tri = ortho_matrix[mask].abs()

    return {
        'mean_orthogonality': float(upper_tri.mean().item()),
        'max_orthogonality': float(upper_tri.max().item()),
        'min_orthogonality': float(upper_tri.min().item()),
        'std_orthogonality': float(upper_tri.std().item()),
        'median_orthogonality': float(upper_tri.median().item())
    }


def format_metrics_for_logging(metrics: Dict[str, float]) -> str:
    """
    Format orthogonality metrics for logging.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Formatted string for logging
    """
    parts = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            parts.append(f"{name}={value:.4f}")
        else:
            parts.append(f"{name}={value}")

    return " | ".join(parts)


def log_orthogonality_to_mlflow(
    layer_representations: Union[torch.Tensor, List[torch.Tensor]],
    step: int,
    prefix: str = "orthogonality",
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compute and log orthogonality metrics to MLflow.

    Args:
        layer_representations: Layer outputs [B, D, L] or List[Tensor]
        step: Training step for logging
        prefix: Prefix for metric names
        normalize: Whether to L2 normalize before computing

    Returns:
        Dictionary of logged metrics
    """
    try:
        import mlflow
    except ImportError:
        # MLflow not available, just return metrics
        return compute_orthogonality_statistics(layer_representations, normalize)

    # Compute statistics
    stats = compute_orthogonality_statistics(layer_representations, normalize)
    layerwise = compute_layerwise_orthogonality(layer_representations, normalize)

    # Log to MLflow
    for name, value in stats.items():
        mlflow.log_metric(f"{prefix}/{name}", value, step=step)

    for name, value in layerwise.items():
        mlflow.log_metric(f"{prefix}/{name}", value, step=step)

    # Combine and return
    all_metrics = {**stats, **layerwise}
    return all_metrics


def compute_condition_number(
    layer_representations: Union[torch.Tensor, List[torch.Tensor]],
    normalize: bool = True
) -> float:
    """
    Compute condition number of the layer similarity matrix.

    Higher condition number indicates more redundant layers.

    Args:
        layer_representations: Layer outputs [B, D, L] or List[Tensor]
        normalize: Whether to L2 normalize before computing

    Returns:
        Condition number (1 = perfectly orthogonal, higher = more redundant)
    """
    from .orthogonal_constraint import OrthogonalConstraint

    constraint = OrthogonalConstraint(normalize=normalize)
    ortho_matrix = constraint.compute_orthogonality_matrix(layer_representations)

    # Compute condition number (ratio of largest to smallest eigenvalue)
    try:
        eigenvalues = torch.linalg.eigvalsh(ortho_matrix)
        eigenvalues = eigenvalues.clamp(min=1e-8)  # Avoid division by zero
        cond_number = eigenvalues.max() / eigenvalues.min()
        return float(cond_number.item())
    except Exception:
        # If eigenvalue computation fails, return infinity
        return float('inf')


def visualize_orthogonality_matrix(
    layer_representations: Union[torch.Tensor, List[torch.Tensor]],
    layer_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    normalize: bool = True
) -> Optional[np.ndarray]:
    """
    Visualize the orthogonality matrix as a heatmap.

    Args:
        layer_representations: Layer outputs [B, D, L] or List[Tensor]
        layer_names: Names for each layer (optional)
        save_path: Path to save the figure (optional)
        normalize: Whether to L2 normalize before computing

    Returns:
        Orthogonality matrix as numpy array (if matplotlib available)
    """
    from .orthogonal_constraint import OrthogonalConstraint

    constraint = OrthogonalConstraint(normalize=normalize)
    ortho_matrix = constraint.compute_orthogonality_matrix(layer_representations)
    ortho_np = ortho_matrix.cpu().numpy()

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        L = ortho_np.shape[0]

        if layer_names is None:
            layer_names = [f"Layer {i}" for i in range(L)]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            ortho_np,
            xticklabels=layer_names,
            yticklabels=layer_names,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt='.2f',
            square=True
        )
        plt.title('Layer Orthogonality Matrix')
        plt.xlabel('Layer')
        plt.ylabel('Layer')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            plt.close()

    except ImportError:
        # Matplotlib not available
        pass

    return ortho_np
