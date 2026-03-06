"""Threshold optimization strategies.

Provides functions and classes for selecting an optimal decision
threshold using inner-fold out-of-fold predictions.  Two strategies are
supported:

* **Fold-specific** -- Optimise a threshold independently within each
  inner fold, then average across folds.
* **Pooled** -- Pool all inner out-of-fold predictions and optimise a
  single threshold.

See Also
--------
nestkit.thresholding.criteria : Built-in criterion functions.
nestkit.thresholding.results.ThresholdResult : Result container.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from nestkit._validation import extract_positive_proba
from nestkit.thresholding.results import ThresholdResult


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    criterion_fn,
) -> tuple[float, float]:
    """Find the threshold that maximises a criterion via grid search.

    Evaluates the criterion function over 991 evenly spaced thresholds
    in [0.01, 0.99] and returns the threshold with the highest value.

    Parameters
    ----------
    y_true : numpy.ndarray of shape (n_samples,)
        True binary labels (0 or 1).
    y_proba : numpy.ndarray of shape (n_samples,) or (n_samples, 2)
        Predicted probabilities.  If 2-D, the positive-class column is
        extracted automatically.
    criterion_fn : callable
        A function with signature
        ``(y_true, y_proba_1d, threshold) -> float`` that returns the
        criterion value to maximise.

    Returns
    -------
    optimal_threshold : float
        The threshold that maximises ``criterion_fn``.
    criterion_value_at_optimum : float
        The criterion value at the optimal threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from nestkit.thresholding.strategies import optimize_threshold
    >>> from nestkit.thresholding.criteria import youden_j
    >>> t, v = optimize_threshold(
    ...     np.array([0, 0, 1, 1]),
    ...     np.array([0.1, 0.4, 0.6, 0.9]),
    ...     youden_j,
    ... )  # doctest: +SKIP
    >>> 0.4 < t < 0.6  # doctest: +SKIP
    True
    """
    p = extract_positive_proba(y_proba)
    thresholds = np.linspace(0.01, 0.99, 991)
    scores = np.array([criterion_fn(y_true, p, t) for t in thresholds])
    best_idx = np.argmax(scores)
    return float(thresholds[best_idx]), float(scores[best_idx])


def compute_threshold_sensitivity(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    criterion_fn,
) -> pd.DataFrame:
    """Compute classification metrics across a full threshold grid.

    Evaluates the criterion function and several standard classification
    metrics (sensitivity, specificity, precision, recall, F1) at 991
    thresholds in [0.01, 0.99].

    Parameters
    ----------
    y_true : numpy.ndarray of shape (n_samples,)
        True binary labels (0 or 1).
    y_proba : numpy.ndarray of shape (n_samples,) or (n_samples, 2)
        Predicted probabilities.
    criterion_fn : callable
        Criterion function with signature
        ``(y_true, y_proba_1d, threshold) -> float``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with 991 rows and columns ``threshold``,
        ``criterion_value``, ``sensitivity``, ``specificity``,
        ``precision``, ``recall``, ``f1``.

    Examples
    --------
    >>> from nestkit.thresholding.strategies import compute_threshold_sensitivity
    >>> from nestkit.thresholding.criteria import youden_j
    >>> df = compute_threshold_sensitivity(
    ...     np.array([0, 0, 1, 1]),
    ...     np.array([0.1, 0.4, 0.6, 0.9]),
    ...     youden_j,
    ... )  # doctest: +SKIP
    >>> df.shape  # doctest: +SKIP
    (991, 7)

    See Also
    --------
    optimize_threshold : Select the single best threshold.
    """
    p = extract_positive_proba(y_proba)
    thresholds = np.linspace(0.01, 0.99, 991)
    rows = []

    for t in thresholds:
        y_pred = (p >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        rows.append(
            {
                "threshold": t,
                "criterion_value": criterion_fn(y_true, p, t),
                "sensitivity": sens,
                "specificity": spec,
                "precision": precision_score(y_true, y_pred, zero_division=0.0),
                "recall": recall_score(y_true, y_pred, zero_division=0.0),
                "f1": f1_score(y_true, y_pred, zero_division=0.0),
            }
        )

    return pd.DataFrame(rows)


class FoldSpecificThreshold:
    """Optimise threshold independently per inner fold, then average.

    This strategy finds the optimal threshold within each inner
    cross-validation fold and uses their mean as the final threshold
    applied to the outer test set.  The standard deviation of per-fold
    thresholds serves as a stability diagnostic.

    See Also
    --------
    PooledThreshold : Alternative strategy that pools all inner OOF data.
    """

    @staticmethod
    def optimize(
        oof_y_true_per_fold: list[np.ndarray],
        oof_proba_per_fold: list[np.ndarray],
        criterion_fn,
        criterion_name: str = "",
    ) -> ThresholdResult:
        """Run fold-specific threshold optimisation.

        Parameters
        ----------
        oof_y_true_per_fold : list of numpy.ndarray
            True labels for each inner fold's out-of-fold predictions.
        oof_proba_per_fold : list of numpy.ndarray
            Predicted probabilities for each inner fold's out-of-fold
            predictions.
        criterion_fn : callable
            Criterion function with signature
            ``(y_true, y_proba_1d, threshold) -> float``.
        criterion_name : str, optional
            Human-readable name for the criterion (stored in the
            result).

        Returns
        -------
        ThresholdResult
            Result with ``strategy="fold_specific"``,
            ``fold_thresholds`` populated, and
            ``fold_threshold_std`` indicating stability.

        Examples
        --------
        >>> from nestkit.thresholding.strategies import FoldSpecificThreshold
        >>> from nestkit.thresholding.criteria import youden_j
        >>> result = FoldSpecificThreshold.optimize(
        ...     oof_y_true_per_fold=[y_fold1, y_fold2],
        ...     oof_proba_per_fold=[p_fold1, p_fold2],
        ...     criterion_fn=youden_j,
        ...     criterion_name="youden_j",
        ... )  # doctest: +SKIP
        >>> result.optimal_threshold  # doctest: +SKIP
        0.48
        """
        fold_thresholds = []
        fold_criterion_values = []

        for y, p in zip(oof_y_true_per_fold, oof_proba_per_fold):
            t, v = optimize_threshold(y, p, criterion_fn)
            fold_thresholds.append(t)
            fold_criterion_values.append(v)

        fold_thresholds_arr = np.array(fold_thresholds)
        optimal = float(np.mean(fold_thresholds_arr))

        # Compute sensitivity at optimal threshold using pooled data
        pooled_y = np.concatenate(oof_y_true_per_fold)
        pooled_p = np.concatenate(oof_proba_per_fold)
        sensitivity_df = compute_threshold_sensitivity(pooled_y, pooled_p, criterion_fn)

        return ThresholdResult(
            strategy="fold_specific",
            optimal_threshold=optimal,
            criterion_name=criterion_name,
            criterion_value_at_optimum=float(np.mean(fold_criterion_values)),
            fold_thresholds=fold_thresholds_arr,
            fold_threshold_std=float(np.std(fold_thresholds_arr, ddof=1)),
            threshold_sensitivity=sensitivity_df,
        )


class PooledThreshold:
    """Pool all inner OOF predictions and optimise a single threshold.

    This strategy concatenates the out-of-fold predictions from all
    inner folds and searches for the threshold that maximises the
    criterion on the pooled data.  This can yield a more stable
    threshold than the fold-specific strategy, at the cost of not
    capturing per-fold variability.

    See Also
    --------
    FoldSpecificThreshold : Alternative strategy that averages per-fold
        thresholds.
    """

    @staticmethod
    def optimize(
        oof_y_true_per_fold: list[np.ndarray],
        oof_proba_per_fold: list[np.ndarray],
        criterion_fn,
        criterion_name: str = "",
    ) -> ThresholdResult:
        """Run pooled threshold optimisation.

        Parameters
        ----------
        oof_y_true_per_fold : list of numpy.ndarray
            True labels for each inner fold's out-of-fold predictions.
        oof_proba_per_fold : list of numpy.ndarray
            Predicted probabilities for each inner fold's out-of-fold
            predictions.
        criterion_fn : callable
            Criterion function with signature
            ``(y_true, y_proba_1d, threshold) -> float``.
        criterion_name : str, optional
            Human-readable name for the criterion.

        Returns
        -------
        ThresholdResult
            Result with ``strategy="pooled"``, ``fold_thresholds=None``,
            and ``fold_threshold_std=None``.

        Examples
        --------
        >>> from nestkit.thresholding.strategies import PooledThreshold
        >>> from nestkit.thresholding.criteria import youden_j
        >>> result = PooledThreshold.optimize(
        ...     oof_y_true_per_fold=[y_fold1, y_fold2],
        ...     oof_proba_per_fold=[p_fold1, p_fold2],
        ...     criterion_fn=youden_j,
        ...     criterion_name="youden_j",
        ... )  # doctest: +SKIP
        >>> result.optimal_threshold  # doctest: +SKIP
        0.50
        """
        pooled_y = np.concatenate(oof_y_true_per_fold)
        pooled_p = np.concatenate(oof_proba_per_fold)

        optimal, criterion_value = optimize_threshold(pooled_y, pooled_p, criterion_fn)
        sensitivity_df = compute_threshold_sensitivity(pooled_y, pooled_p, criterion_fn)

        return ThresholdResult(
            strategy="pooled",
            optimal_threshold=optimal,
            criterion_name=criterion_name,
            criterion_value_at_optimum=criterion_value,
            fold_thresholds=None,
            fold_threshold_std=None,
            threshold_sensitivity=sensitivity_df,
        )
