"""Input validation helpers for nestkit parameters.

These functions are used internally by :class:`~nestkit.NestedCVClassifier`
and :class:`~nestkit.NestedCVRegressor` to validate user-provided
configuration before the nested CV procedure begins.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_threshold_params(
    threshold_strategy: str | None,
    threshold_criterion: str | Any,
    cost_matrix: Any | None,
    min_recall: float | None,
) -> None:
    """Validate threshold optimization parameters.

    Parameters
    ----------
    threshold_strategy : str or None
        One of ``'fold_specific'``, ``'pooled'``, or ``None``.
    threshold_criterion : str or callable
        Threshold selection criterion name or callable.
    cost_matrix : array-like or None
        Cost matrix required for ``'cost'`` criterion.
    min_recall : float or None
        Minimum recall required for ``'precision_at_recall'`` criterion.

    Raises
    ------
    ValueError
        If any parameter combination is invalid.
    """
    valid_strategies = {None, "fold_specific", "pooled"}
    if threshold_strategy not in valid_strategies:
        raise ValueError(
            f"threshold_strategy must be one of {valid_strategies}, got '{threshold_strategy}'"
        )

    valid_criteria = {"youden", "f_beta", "cost", "balanced_accuracy", "precision_at_recall"}
    if isinstance(threshold_criterion, str) and threshold_criterion not in valid_criteria:
        raise ValueError(
            f"threshold_criterion must be one of {valid_criteria} or callable, "
            f"got '{threshold_criterion}'"
        )

    if threshold_criterion == "cost" and cost_matrix is None:
        raise ValueError("cost_matrix is required when threshold_criterion='cost'")

    if threshold_criterion == "precision_at_recall" and min_recall is None:
        raise ValueError("min_recall is required when threshold_criterion='precision_at_recall'")


def validate_calibration_method(method: str | None) -> None:
    """Validate the calibration method parameter.

    Parameters
    ----------
    method : str or None
        One of ``'sigmoid'``, ``'isotonic'``, ``'beta'``,
        ``'venn_abers'``, or ``None``.

    Raises
    ------
    ValueError
        If ``method`` is not a recognized calibration method.
    """
    valid = {None, "sigmoid", "isotonic", "beta", "venn_abers"}
    if method not in valid:
        raise ValueError(f"calibration_method must be one of {valid}, got '{method}'")


def ensure_2d_proba(y_proba: np.ndarray) -> np.ndarray:
    """Ensure probability array is 2D ``(n_samples, n_classes)``.

    Parameters
    ----------
    y_proba : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Probability predictions. If 1-D, interpreted as positive-class
        probabilities for a binary problem.

    Returns
    -------
    ndarray of shape (n_samples, 2) or (n_samples, n_classes)
        Two-dimensional probability array.
    """
    if y_proba.ndim == 1:
        return np.column_stack([1 - y_proba, y_proba])
    return y_proba


def extract_positive_proba(y_proba: np.ndarray) -> np.ndarray:
    """Extract positive-class probabilities from a probability array.

    Parameters
    ----------
    y_proba : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Probability predictions. If 2-D, the second column (index 1) is
        returned.

    Returns
    -------
    ndarray of shape (n_samples,)
        Positive-class probabilities.
    """
    if y_proba.ndim == 2:
        return y_proba[:, 1]
    return y_proba
