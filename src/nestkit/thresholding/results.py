"""Threshold optimization result container.

Provides the ``ThresholdResult`` dataclass that captures the output of
a threshold optimisation procedure for a single outer fold in nested
cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ThresholdResult:
    """Threshold optimization results for a single outer fold.

    Stores the optimal threshold, the criterion used, per-inner-fold
    thresholds (for the fold-specific strategy), and a full
    threshold-sensitivity grid for downstream analysis and plotting.

    Attributes
    ----------
    strategy : str
        Threshold selection strategy: ``"fold_specific"`` (optimise per
        inner fold then average) or ``"pooled"`` (optimise on pooled
        inner out-of-fold predictions).
    optimal_threshold : float
        The decision threshold applied to the outer test set for
        evaluation.
    criterion_name : str
        Human-readable name of the optimisation criterion (e.g.,
        ``"youden_j"``, ``"f_1.0"``).
    criterion_value_at_optimum : float
        Value of the criterion function at ``optimal_threshold``.
    fold_thresholds : numpy.ndarray or None
        Per-inner-fold optimal thresholds.  Only populated for the
        ``"fold_specific"`` strategy; ``None`` for ``"pooled"``.
    fold_threshold_std : float or None
        Standard deviation of ``fold_thresholds``, serving as a
        stability indicator for the fold-specific strategy.  ``None``
        for ``"pooled"``.
    threshold_sensitivity : pandas.DataFrame
        Full threshold-sensitivity grid with columns ``threshold``,
        ``criterion_value``, ``sensitivity``, ``specificity``,
        ``precision``, ``recall``, ``f1``.  Useful for plotting
        threshold-performance curves.

    See Also
    --------
    nestkit.thresholding.strategies.FoldSpecificThreshold :
        Produces ``ThresholdResult`` with ``strategy="fold_specific"``.
    nestkit.thresholding.strategies.PooledThreshold :
        Produces ``ThresholdResult`` with ``strategy="pooled"``.
    nestkit.thresholding.criteria : Built-in criterion functions.

    Examples
    --------
    >>> result.optimal_threshold  # doctest: +SKIP
    0.42
    >>> result.threshold_sensitivity.head()  # doctest: +SKIP
       threshold  criterion_value  sensitivity  specificity  ...
    """

    strategy: str
    optimal_threshold: float
    criterion_name: str
    criterion_value_at_optimum: float
    fold_thresholds: np.ndarray | None = None
    fold_threshold_std: float | None = None
    threshold_sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)
