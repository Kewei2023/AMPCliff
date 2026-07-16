"""Shared Ridge probe utilities for DC validation property decoding experiments."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

RIDGE_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]


def bootstrap_spearman_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 3:
        rho, _ = spearmanr(y_true, y_pred)
        val = float(rho) if rho == rho else 0.0
        return val, val
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=n)
        rho, _ = spearmanr(y_true[pick], y_pred[pick])
        if rho == rho:
            stats.append(float(rho))
    if not stats:
        rho, _ = spearmanr(y_true, y_pred)
        val = float(rho) if rho == rho else 0.0
        return val, val
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


def bootstrap_delta_spearman_ci(
    y_true: np.ndarray,
    pred_c0: np.ndarray,
    pred_c1: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap CI for rho(C0) - rho(C1) on the same test samples."""
    rho0, _ = spearmanr(y_true, pred_c0)
    rho1, _ = spearmanr(y_true, pred_c1)
    delta = float(rho0 - rho1) if rho0 == rho0 and rho1 == rho1 else np.nan

    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 3:
        val = float(delta) if delta == delta else 0.0
        return float(delta) if delta == delta else np.nan, val, val

    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=n)
        y_b = y_true[pick]
        r0, _ = spearmanr(y_b, pred_c0[pick])
        r1, _ = spearmanr(y_b, pred_c1[pick])
        if r0 == r0 and r1 == r1:
            deltas.append(float(r0 - r1))
    if not deltas:
        val = float(delta) if delta == delta else 0.0
        return float(delta) if delta == delta else np.nan, val, val
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return float(delta) if delta == delta else np.nan, lo, hi


def select_alpha_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    alphas: Sequence[float],
) -> Tuple[float, Ridge]:
    best_alpha = float(alphas[0])
    best_score = -np.inf
    best_model: Optional[Ridge] = None
    for alpha in alphas:
        model = Ridge(alpha=float(alpha))
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        rho, _ = spearmanr(y_valid, pred)
        score = float(rho) if rho == rho else -np.inf
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_model = model
    assert best_model is not None
    return best_alpha, best_model


def run_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    bootstrap_seed: int = 0,
    return_predictions: bool = False,
) -> Dict[str, object]:
    _, model = select_alpha_ridge(X_train, y_train, X_valid, y_valid, alphas)
    model.fit(
        np.concatenate([X_train, X_valid], axis=0),
        np.concatenate([y_train, y_valid], axis=0),
    )
    pred = model.predict(X_test)
    rho, _ = spearmanr(y_test, pred)
    ci_lo, ci_hi = bootstrap_spearman_ci(y_test, pred, seed=bootstrap_seed)
    out: Dict[str, object] = {
        "spearman": float(rho) if rho == rho else np.nan,
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "spearman_ci_lo": ci_lo,
        "spearman_ci_hi": ci_hi,
    }
    if return_predictions:
        out["y_test"] = y_test
        out["y_pred"] = pred
    return out
