"""CV+ Mondrian conformal prediction intervals for regression."""

from __future__ import annotations

import warnings

import numpy as np

from nestkit.conformal.results import RegressorConformalResult


class MondrianRegressorConformal:
    """Mondrian conformal prediction intervals conditioned on predicted value.

    Bins OOF predictions into equal-frequency groups and computes
    per-bin residual quantiles, yielding tighter intervals in
    easy-to-predict regions and wider intervals elsewhere.
    """

    @staticmethod
    def fit(
        oof_predictions: np.ndarray,
        oof_residuals: np.ndarray,
        alpha: float = 0.05,
        n_bins: int = 5,
        min_bin_size: int = 20,
    ) -> RegressorConformalResult:
        """Compute per-bin residual quantiles from OOF data.

        Parameters
        ----------
        oof_predictions : ndarray of shape (n_cal,)
            Out-of-fold point predictions.
        oof_residuals : ndarray of shape (n_cal,)
            Signed residuals ``y_true - y_pred``.
        alpha : float
            Significance level (default 0.05 for 95% coverage).
        n_bins : int
            Number of equal-frequency bins for Mondrian conditioning.
        min_bin_size : int
            Minimum calibration samples per bin; smaller bins are merged.
            The effective floor is raised to
            :func:`min_calibration_size(alpha) <min_calibration_size>` when
            that is larger, since a smaller bin cannot produce a finite
            interval that reaches the target coverage.

        Returns
        -------
        RegressorConformalResult

        Notes
        -----
        A bin holding fewer than ``min_calibration_size(alpha)`` residuals
        has no valid two-sided order statistic, so bins are merged until
        every one of them clears that floor.  With the default
        ``alpha=0.05`` the floor is 39, not the nominal ``min_bin_size=20``:
        a 20-point bin only reaches about 0.90 coverage against a 0.95
        target.
        """
        n_cal = len(oof_predictions)

        effective_min = max(min_bin_size, min_calibration_size(alpha))

        max_bins = max(1, n_cal // effective_min)
        if n_bins > max_bins:
            warnings.warn(
                f"n_bins={n_bins} would give < {effective_min} samples per bin "
                f"with {n_cal} calibration points; reducing to n_bins={max_bins}. "
                f"At alpha={alpha:g} a valid two-sided conformal interval needs "
                f"at least {min_calibration_size(alpha)} residuals per bin.",
                UserWarning,
                stacklevel=2,
            )
            n_bins = max_bins

        fallback_quantiles = _corrected_residual_quantiles(oof_residuals, alpha)

        if n_bins == 1:
            return RegressorConformalResult(
                alpha=alpha,
                n_bins=1,
                bin_edges=np.array([-np.inf, np.inf]),
                bin_quantiles=[fallback_quantiles],
                bin_counts=np.array([n_cal]),
                fallback_quantiles=fallback_quantiles,
            )

        quantile_fracs = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(oof_predictions, quantile_fracs)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        assignments = np.digitize(oof_predictions, bin_edges[1:-1], right=False)
        bin_edges, assignments = _merge_small_bins(bin_edges, assignments, n_bins, effective_min)
        final_n_bins = len(bin_edges) - 1

        bin_quantiles = []
        bin_counts = np.empty(final_n_bins, dtype=int)
        for b in range(final_n_bins):
            mask = assignments == b
            n_b = int(mask.sum())
            bin_counts[b] = n_b
            if n_b == 0:
                bin_quantiles.append(fallback_quantiles)
                continue
            resid_b = oof_residuals[mask]
            bin_quantiles.append(_corrected_residual_quantiles(resid_b, alpha))

        return RegressorConformalResult(
            alpha=alpha,
            n_bins=final_n_bins,
            bin_edges=bin_edges,
            bin_quantiles=bin_quantiles,
            bin_counts=bin_counts,
            fallback_quantiles=fallback_quantiles,
        )

    @staticmethod
    def predict(
        test_predictions: np.ndarray,
        conformal_result: RegressorConformalResult,
    ) -> dict:
        """Generate per-bin prediction intervals for test predictions.

        Parameters
        ----------
        test_predictions : ndarray of shape (n_test,)
            Point predictions for test data.
        conformal_result : RegressorConformalResult
            Result from :meth:`fit`.

        Returns
        -------
        dict
            ``lower``: ndarray of lower bounds.
            ``upper``: ndarray of upper bounds.
            ``bin_assignments``: ndarray of int bin indices.
        """
        edges = conformal_result.bin_edges
        n_bins = conformal_result.n_bins

        assignments = np.digitize(test_predictions, edges[1:-1], right=False)
        assignments = np.clip(assignments, 0, n_bins - 1)

        test_predictions = np.asarray(test_predictions, dtype=float)
        lower = np.empty(test_predictions.shape[0], dtype=float)
        upper = np.empty(test_predictions.shape[0], dtype=float)

        for b in range(n_bins):
            mask = assignments == b
            if not mask.any():
                continue
            q_lo, q_hi = conformal_result.bin_quantiles[b]
            lower[mask] = test_predictions[mask] + q_lo
            upper[mask] = test_predictions[mask] + q_hi

        return {
            "lower": lower,
            "upper": upper,
            "bin_assignments": assignments,
        }


def min_calibration_size(alpha: float) -> int:
    """Smallest calibration set that supports a valid two-sided interval.

    A two-sided conformal interval at level ``alpha`` reads off the
    ``floor((alpha/2)(n+1))``-th and ``ceil((1-alpha/2)(n+1))``-th order
    statistics of the calibration residuals.  Both indices exist only when

    .. math::

        n \\ge \\frac{2 - \\alpha}{\\alpha}

    which is 39 at the default ``alpha=0.05`` and 19 at ``alpha=0.1``.
    Below that, at least one bound is unbounded and the interval cannot be
    finite without losing the coverage guarantee.

    Parameters
    ----------
    alpha : float
        Significance level in ``(0, 1)``.

    Returns
    -------
    int
        Minimum number of calibration points per bin.
    """
    return int(np.ceil((2.0 - alpha) / alpha))


def _corrected_residual_quantiles(residuals: np.ndarray, alpha: float) -> tuple[float, float]:
    """Exact order-statistic residual quantiles with finite-sample correction.

    Returns exact order statistics from the sorted residual array rather
    than interpolating with ``np.quantile``.  This matches the approach used
    in :class:`MondrianClassifierConformal` and is required for the formal
    conformal coverage guarantee.  A bound that the sample cannot determine
    is returned as ``-inf``/``inf`` instead of being clipped to the extreme
    residual, which would silently undercover.
    """
    n = len(residuals)
    if n == 0:
        return -np.inf, np.inf

    sorted_resid = np.sort(residuals)
    k_lo = int(np.floor((alpha / 2) * (n + 1)))
    k_hi = int(np.ceil((1 - alpha / 2) * (n + 1)))

    lo = -np.inf if k_lo < 1 else float(sorted_resid[k_lo - 1])
    hi = np.inf if k_hi > n else float(sorted_resid[k_hi - 1])

    if not np.isfinite(lo) or not np.isfinite(hi):
        warnings.warn(
            f"{n} calibration residuals are not enough for a finite two-sided "
            f"conformal interval at alpha={alpha:g}, which needs at least "
            f"{min_calibration_size(alpha)}. Returning an unbounded side rather "
            "than a narrower interval that would not reach the target coverage.",
            UserWarning,
            stacklevel=2,
        )
    return lo, hi


def _merge_small_bins(
    bin_edges: np.ndarray,
    assignments: np.ndarray,
    n_bins: int,
    min_bin_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge bins with fewer than ``min_bin_size`` samples into neighbours."""
    while True:
        unique_bins = np.arange(len(bin_edges) - 1)
        counts = np.array([int((assignments == b).sum()) for b in unique_bins])
        small = np.where(counts < min_bin_size)[0]

        if len(small) == 0 or len(unique_bins) <= 1:
            break

        victim = small[np.argmin(counts[small])]
        current_n_bins = len(unique_bins)

        if victim == 0:
            merge_with = 1
        elif victim == current_n_bins - 1:
            merge_with = current_n_bins - 2
        else:
            merge_with = victim - 1 if counts[victim - 1] <= counts[victim + 1] else victim + 1

        lo, hi = sorted([victim, merge_with])
        bin_edges = np.delete(bin_edges, lo + 1)
        assignments[assignments == hi] = lo
        assignments[assignments > hi] -= 1

    return bin_edges, assignments
