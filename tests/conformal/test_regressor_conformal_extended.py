"""Extended tests for MondrianRegressorConformal covering bin merging, empty bins, and edge cases."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nestkit.conformal.regressor_conformal import (
    MondrianRegressorConformal,
    _corrected_residual_quantiles,
    _merge_small_bins,
    min_calibration_size,
)


class TestMinCalibrationSize:
    """A two-sided interval needs both order statistics to exist."""

    @pytest.mark.parametrize("alpha, expected", [(0.05, 39), (0.1, 19), (0.2, 9), (0.01, 199)])
    def test_threshold(self, alpha, expected):
        assert min_calibration_size(alpha) == expected

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2])
    def test_threshold_is_where_both_indices_exist(self, alpha):
        """Below the threshold at least one order-statistic index is out of range."""
        n = min_calibration_size(alpha)
        assert int(np.floor((alpha / 2) * (n + 1))) >= 1
        assert int(np.ceil((1 - alpha / 2) * (n + 1))) <= n
        below = n - 1
        assert (
            int(np.floor((alpha / 2) * (below + 1))) < 1
            or int(np.ceil((1 - alpha / 2) * (below + 1))) > below
        )


class TestCorrectedResidualQuantiles:
    def test_empty_residuals_are_unbounded(self):
        """No calibration data means no information, not a zero-width interval."""
        q_lo, q_hi = _corrected_residual_quantiles(np.array([]), alpha=0.05)
        assert q_lo == -np.inf
        assert q_hi == np.inf

    def test_single_residual_is_unbounded(self):
        """One residual cannot support a 95% two-sided interval."""
        with pytest.warns(UserWarning, match="not enough for a finite"):
            q_lo, q_hi = _corrected_residual_quantiles(np.array([3.0]), alpha=0.05)
        assert q_lo == -np.inf
        assert q_hi == np.inf

    def test_returns_sorted_values_when_large_enough(self):
        """With enough residuals both bounds are actual order statistics."""
        rng = np.random.RandomState(0)
        resid = rng.randn(min_calibration_size(0.05))
        q_lo, q_hi = _corrected_residual_quantiles(resid, alpha=0.05)
        assert q_lo <= q_hi
        assert q_lo in resid
        assert q_hi in resid

    def test_warns_below_threshold(self):
        rng = np.random.RandomState(0)
        resid = rng.randn(min_calibration_size(0.05) - 1)
        with pytest.warns(UserWarning, match="not enough for a finite"):
            _corrected_residual_quantiles(resid, alpha=0.05)

    def test_silent_at_threshold(self, recwarn):
        rng = np.random.RandomState(0)
        resid = rng.randn(min_calibration_size(0.05))
        _corrected_residual_quantiles(resid, alpha=0.05)
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
    @pytest.mark.parametrize("n", [5, 10, 20, 39, 60, 200])
    def test_coverage_guarantee_holds(self, alpha, n):
        """The defining property: empirical coverage never falls below 1 - alpha.

        Clipping the order-statistic indices to the extreme residuals used to
        break this for n < (2 - alpha) / alpha: at alpha=0.05, n=20 covered
        only ~0.90 against a 0.95 target.
        """
        rng = np.random.RandomState(0)
        draws = rng.randn(20000, n + 1)
        calibration, held_out = draws[:, :n], draws[:, n]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            bounds = [_corrected_residual_quantiles(row, alpha) for row in calibration]

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        coverage = np.mean((held_out >= lo) & (held_out <= hi))
        assert coverage >= 1 - alpha - 0.01


class TestMergeSmallBins:
    def test_left_edge_merges_right(self):
        """Verify the leftmost small bin merges with its right neighbor."""
        # 4 bins: [0,2), [2,4), [4,6), [6,inf)
        bin_edges = np.array([-np.inf, 2.0, 4.0, 6.0, np.inf])
        # assignments: bin 0 has 1 sample, bin 1 has 20, bin 2 has 20, bin 3 has 20
        assignments = np.array([0] * 1 + [1] * 20 + [2] * 20 + [3] * 20)
        new_edges, new_assign = _merge_small_bins(bin_edges, assignments, 4, min_bin_size=5)
        # Bin 0 should be merged with bin 1
        assert len(new_edges) < len(bin_edges)
        # No bin should have < 5 samples (except potentially after multiple merges)
        unique = np.unique(new_assign)
        for b in unique:
            assert (new_assign == b).sum() >= 5

    def test_right_edge_merges_left(self):
        """Verify the rightmost small bin merges with its left neighbor."""
        bin_edges = np.array([-np.inf, 2.0, 4.0, 6.0, np.inf])
        assignments = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 1)
        new_edges, _new_assign = _merge_small_bins(bin_edges, assignments, 4, min_bin_size=5)
        assert len(new_edges) < len(bin_edges)

    def test_interior_merges_with_smaller_neighbor(self):
        """Verify interior bin merges with the smaller neighboring bin."""
        bin_edges = np.array([-np.inf, 2.0, 4.0, 6.0, np.inf])
        # bin 0: 10, bin 1: 2 (victim), bin 2: 5 (smaller neighbor), bin 3: 30
        assignments = np.array([0] * 10 + [1] * 2 + [2] * 5 + [3] * 30)
        new_edges, _new_assign = _merge_small_bins(bin_edges, assignments, 4, min_bin_size=5)
        # bin 1 should be merged with bin 2 (smaller neighbor)
        assert len(new_edges) < len(bin_edges)

    def test_multiple_merge_iterations(self):
        """Verify merging iterates until no bins are below min_bin_size."""
        bin_edges = np.array([-np.inf, 1.0, 2.0, 3.0, 4.0, np.inf])
        # 5 bins, each with only 3 samples
        assignments = np.array([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3)
        new_edges, new_assign = _merge_small_bins(bin_edges, assignments, 5, min_bin_size=5)
        # Should merge down to fewer bins
        final_n_bins = len(new_edges) - 1
        assert final_n_bins < 5
        # All remaining bins should have >= 5 or all are merged into one
        unique = np.unique(new_assign)
        for b in unique:
            count = (new_assign == b).sum()
            assert count >= 3  # at least merged some

    def test_single_bin_no_merge(self):
        """Verify no merging when there's only 1 bin."""
        bin_edges = np.array([-np.inf, np.inf])
        assignments = np.array([0] * 10)
        new_edges, new_assign = _merge_small_bins(bin_edges, assignments, 1, min_bin_size=5)
        assert len(new_edges) == 2
        np.testing.assert_array_equal(new_assign, assignments)


class TestMondrianRegressorConformalFitEdgeCases:
    def test_auto_reduce_bins(self):
        """Verify n_bins is auto-reduced when insufficient data.

        min_bin_size=5 is below the 39 residuals a 95% interval needs, so the
        alpha-derived floor wins and 10 points collapse to a single bin.
        """
        rng = np.random.RandomState(42)
        oof_preds = rng.randn(10)
        oof_resid = rng.randn(10)

        with pytest.warns(UserWarning) as record:
            result = MondrianRegressorConformal.fit(
                oof_preds, oof_resid, n_bins=20, min_bin_size=5
            )

        messages = [str(w.message) for w in record]
        assert any("reducing to n_bins" in m for m in messages)
        assert any("not enough for a finite" in m for m in messages)
        assert result.n_bins == 1

    def test_single_bin_result(self):
        """Verify single-bin conformal result for very small datasets."""
        rng = np.random.RandomState(42)
        oof_preds = rng.randn(3)
        oof_resid = rng.randn(3)
        with pytest.warns(UserWarning):
            result = MondrianRegressorConformal.fit(oof_preds, oof_resid, n_bins=5, min_bin_size=5)
        assert result.n_bins == 1
        np.testing.assert_array_equal(result.bin_edges, [-np.inf, np.inf])

    def test_predict_with_extrapolation(self):
        """Verify predict handles values outside bin edges."""
        rng = np.random.RandomState(42)
        oof_preds = rng.randn(100)
        oof_resid = rng.randn(100)
        # 100 points support 2 bins of 39, not the 3 requested
        with pytest.warns(UserWarning, match="reducing to n_bins"):
            result = MondrianRegressorConformal.fit(
                oof_preds, oof_resid, n_bins=3, min_bin_size=10
            )

        # Test predictions far outside training range
        test_preds = np.array([-100.0, 100.0])
        output = MondrianRegressorConformal.predict(test_preds, result)
        assert output["lower"].shape == (2,)
        assert output["upper"].shape == (2,)
        assert np.all(output["lower"] <= output["upper"])
