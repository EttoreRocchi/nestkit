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

        Returns
        -------
        RegressorConformalResult
        """
        n_cal = len(oof_predictions)

        # Auto-reduce n_bins if insufficient data
        max_bins = max(1, n_cal // min_bin_size)
        if n_bins > max_bins:
            warnings.warn(
                f"n_bins={n_bins} would give < {min_bin_size} samples per bin "
                f"with {n_cal} calibration points; reducing to n_bins={max_bins}.",
                UserWarning,
                stacklevel=2,
            )
            n_bins = max_bins

        # Global fallback quantiles (exact order statistics, finite-sample corrected)
        fallback_quantiles = _corrected_residual_quantiles(oof_residuals, alpha)

        if n_bins == 1:
            # Single bin = global conformal
            return RegressorConformalResult(
                alpha=alpha,
                n_bins=1,
                bin_edges=np.array([-np.inf, np.inf]),
                bin_quantiles=[fallback_quantiles],
                bin_counts=np.array([n_cal]),
                fallback_quantiles=fallback_quantiles,
            )

        # Equal-frequency bin edges from quantiles of OOF predictions
        quantile_fracs = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(oof_predictions, quantile_fracs)
        # Ensure extreme edges capture all data
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Assign calibration points to bins
        assignments = np.digitize(oof_predictions, bin_edges[1:-1], right=False)
        # digitize returns 0..n_bins-1 with this setup

        # Merge small bins with nearest neighbor
        bin_edges, assignments = _merge_small_bins(bin_edges, assignments, n_bins, min_bin_size)
        final_n_bins = len(bin_edges) - 1

        # Compute per-bin quantiles
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

        # Assign to bins (clamp to valid range)
        assignments = np.digitize(test_predictions, edges[1:-1], right=False)
        assignments = np.clip(assignments, 0, n_bins - 1)

        lower = np.empty_like(test_predictions)
        upper = np.empty_like(test_predictions)

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


def _corrected_quantile_levels(n: int, alpha: float) -> tuple[float, float]:
    """Finite-sample corrected quantile levels for conformal intervals.

    .. deprecated::
        Prefer :func:`_corrected_residual_quantiles` which uses exact order
        statistics instead of ``np.quantile`` interpolation.
    """
    if n == 0:
        return 0.0, 1.0
    q_lo = max(0.0, np.floor((alpha / 2) * (n + 1)) / n)
    q_hi = min(1.0, np.ceil((1 - alpha / 2) * (n + 1)) / n)
    return float(q_lo), float(q_hi)


def _corrected_residual_quantiles(residuals: np.ndarray, alpha: float) -> tuple[float, float]:
    """Exact order-statistic residual quantiles with finite-sample correction.

    Uses the same correction as :func:`_corrected_quantile_levels` but
    returns exact order statistics from the sorted residual array instead
    of relying on ``np.quantile`` interpolation.  This matches the approach
    used in :class:`MondrianClassifierConformal` and is required for formal
    conformal coverage guarantees.
    """
    n = len(residuals)
    if n == 0:
        return 0.0, 0.0
    sorted_resid = np.sort(residuals)
    # Lower: floor-based index (conservative lower bound)
    k_lo = int(np.floor((alpha / 2) * (n + 1))) - 1  # 0-indexed
    k_lo = max(0, min(k_lo, n - 1))
    # Upper: ceil-based index (conservative upper bound)
    k_hi = int(np.ceil((1 - alpha / 2) * (n + 1))) - 1  # 0-indexed
    k_hi = max(0, min(k_hi, n - 1))
    return float(sorted_resid[k_lo]), float(sorted_resid[k_hi])


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

        # Pick the smallest bin
        victim = small[np.argmin(counts[small])]
        current_n_bins = len(unique_bins)

        if victim == 0:
            merge_with = 1
        elif victim == current_n_bins - 1:
            merge_with = current_n_bins - 2
        else:
            # Merge with the neighbour that has fewer samples (balance)
            merge_with = victim - 1 if counts[victim - 1] <= counts[victim + 1] else victim + 1

        # Merge: absorb victim into merge_with
        lo, hi = sorted([victim, merge_with])
        # Remove the interior edge between lo and hi
        bin_edges = np.delete(bin_edges, lo + 1)
        # Reassign: anything in hi becomes lo
        assignments[assignments == hi] = lo
        # Shift higher bin indices down
        assignments[assignments > hi] -= 1

    return bin_edges, assignments
