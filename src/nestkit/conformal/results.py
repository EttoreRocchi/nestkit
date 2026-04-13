"""Conformal prediction result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClassifierConformalResult:
    """Per-class quantile thresholds from CV+ Mondrian conformal calibration.

    Attributes
    ----------
    alpha : float
        Significance level (miscoverage rate).
    qhat_per_class : ndarray of shape (n_classes,)
        Mondrian q-hat threshold for each class.
    n_calibration_per_class : ndarray of shape (n_classes,)
        Number of calibration samples used per class.
    """

    alpha: float
    qhat_per_class: np.ndarray
    n_calibration_per_class: np.ndarray


@dataclass
class RegressorConformalResult:
    """Per-bin residual quantiles from Mondrian regression conformal calibration.

    Attributes
    ----------
    alpha : float
        Significance level (miscoverage rate).
    n_bins : int
        Number of Mondrian bins (after any merging).
    bin_edges : ndarray of shape (n_bins + 1,)
        Bin edges computed from quantiles of OOF predictions.
    bin_quantiles : list of (float, float)
        ``(q_lo, q_hi)`` residual quantile pair per bin.
    bin_counts : ndarray of shape (n_bins,)
        Number of calibration samples in each bin.
    fallback_quantiles : tuple of (float, float)
        Global residual quantiles used for extrapolation.
    """

    alpha: float
    n_bins: int
    bin_edges: np.ndarray
    bin_quantiles: list[tuple[float, float]]
    bin_counts: np.ndarray
    fallback_quantiles: tuple[float, float]
