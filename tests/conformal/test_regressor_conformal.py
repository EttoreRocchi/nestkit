"""Tests for MondrianRegressorConformal -- Mondrian binned prediction intervals."""

import numpy as np
import pytest

from nestkit.conformal.regressor_conformal import (
    MondrianRegressorConformal,
    _corrected_residual_quantiles,
)
from nestkit.conformal.results import RegressorConformalResult


@pytest.fixture()
def regression_data():
    """OOF predictions and residuals with heteroscedastic noise."""
    rng = np.random.RandomState(42)
    n = 300
    preds = rng.uniform(0, 100, size=n)
    # Heteroscedastic: noise proportional to predicted value
    noise_scale = 0.1 * preds + 1
    residuals = rng.normal(0, noise_scale, size=n)
    return preds, residuals


class TestFit:
    def test_basic_result_structure(self, regression_data):
        preds, residuals = regression_data
        result = MondrianRegressorConformal.fit(preds, residuals, alpha=0.05, n_bins=5)
        assert isinstance(result, RegressorConformalResult)
        assert result.n_bins >= 1
        assert len(result.bin_quantiles) == result.n_bins
        assert result.bin_edges.shape == (result.n_bins + 1,)
        assert result.bin_counts.shape == (result.n_bins,)
        assert result.alpha == 0.05

    def test_single_bin_fallback(self, regression_data):
        preds, residuals = regression_data
        result = MondrianRegressorConformal.fit(preds, residuals, n_bins=1)
        assert result.n_bins == 1
        assert result.bin_edges[0] == -np.inf
        assert result.bin_edges[-1] == np.inf

    def test_auto_reduce_bins(self):
        """When n_bins is too large for the data, it gets auto-reduced."""
        rng = np.random.RandomState(0)
        preds = rng.uniform(0, 10, size=30)
        residuals = rng.normal(0, 1, size=30)
        with pytest.warns(UserWarning, match="reducing to n_bins"):
            result = MondrianRegressorConformal.fit(preds, residuals, n_bins=10, min_bin_size=20)
        assert result.n_bins == 1  # 30 / 20 = 1

    def test_bin_counts_sum_to_n(self, regression_data):
        preds, residuals = regression_data
        result = MondrianRegressorConformal.fit(preds, residuals, n_bins=5)
        assert result.bin_counts.sum() == len(preds)

    def test_fallback_quantiles_are_global(self, regression_data):
        preds, residuals = regression_data
        result = MondrianRegressorConformal.fit(preds, residuals, n_bins=3)
        q_lo, q_hi = result.fallback_quantiles
        assert q_lo < 0  # residuals centered around 0
        assert q_hi > 0


class TestPredict:
    def test_output_shapes(self, regression_data):
        preds, residuals = regression_data
        result = MondrianRegressorConformal.fit(preds, residuals, n_bins=3)

        test_preds = np.array([10, 50, 90])
        output = MondrianRegressorConformal.predict(test_preds, result)

        assert output["lower"].shape == (3,)
        assert output["upper"].shape == (3,)
        assert output["bin_assignments"].shape == (3,)

    def test_lower_less_than_upper(self, regression_data):
        preds, residuals = regression_data
        result = MondrianRegressorConformal.fit(preds, residuals, n_bins=3)
        output = MondrianRegressorConformal.predict(preds, result)
        assert np.all(output["lower"] < output["upper"])

    def test_extrapolation_handled(self, regression_data):
        """Test predictions outside the calibration range don't crash."""
        preds, residuals = regression_data
        result = MondrianRegressorConformal.fit(preds, residuals, n_bins=3)
        # Values far outside calibration range
        test_preds = np.array([-1000, 1000])
        output = MondrianRegressorConformal.predict(test_preds, result)
        assert output["lower"].shape == (2,)
        assert np.all(np.isfinite(output["lower"]))
        assert np.all(np.isfinite(output["upper"]))

    def test_different_bins_give_different_widths(self, regression_data):
        """Mondrian intervals should have different widths per bin (heteroscedastic data)."""
        preds, residuals = regression_data
        result = MondrianRegressorConformal.fit(preds, residuals, n_bins=5, min_bin_size=10)

        widths = [q_hi - q_lo for q_lo, q_hi in result.bin_quantiles]
        # With heteroscedastic data, widths should not all be identical
        assert len(set(np.round(widths, 4))) > 1


class TestCoverage:
    @pytest.mark.parametrize("n_bins", [1, 3, 5])
    def test_coverage_near_target(self, n_bins):
        """With enough data, per-bin coverage should be near 1-alpha."""
        rng = np.random.RandomState(42)
        n = 500
        preds = rng.uniform(0, 100, size=n)
        true_values = preds + rng.normal(0, 5, size=n)
        residuals = true_values - preds

        alpha = 0.1
        result = MondrianRegressorConformal.fit(
            preds, residuals, alpha=alpha, n_bins=n_bins, min_bin_size=10
        )
        output = MondrianRegressorConformal.predict(preds, result)

        coverage = np.mean((true_values >= output["lower"]) & (true_values <= output["upper"]))
        assert coverage >= 1 - alpha - 0.05


class TestExactOrderStatistics:
    """Verify that residual quantiles use exact order statistics, not interpolation."""

    def test_quantile_is_exact_data_point(self):
        """The returned quantile values must be actual data points, not interpolated."""
        rng = np.random.RandomState(42)
        residuals = rng.normal(0, 1, size=50)
        alpha = 0.1

        q_lo, q_hi = _corrected_residual_quantiles(residuals, alpha)
        sorted_resid = np.sort(residuals)

        # Both quantiles must be exact values from the data
        assert q_lo in sorted_resid
        assert q_hi in sorted_resid

    def test_matches_classifier_conformal_approach(self):
        """Verify equivalence with the np.sort[k-1] approach used by classifier conformal."""
        rng = np.random.RandomState(0)
        residuals = rng.normal(0, 2, size=100)
        alpha = 0.05

        _q_lo, q_hi = _corrected_residual_quantiles(residuals, alpha)

        # The upper quantile should be the ceil((1-alpha/2)*(n+1))-th order statistic
        n = len(residuals)
        k_hi = int(np.ceil((1 - alpha / 2) * (n + 1)))
        k_hi = min(k_hi, n)
        expected_hi = float(np.sort(residuals)[k_hi - 1])
        assert q_hi == pytest.approx(expected_hi)

    def test_empty_residuals(self):
        """Empty residuals should return (0.0, 0.0)."""
        q_lo, q_hi = _corrected_residual_quantiles(np.array([]), 0.1)
        assert q_lo == 0.0
        assert q_hi == 0.0

    def test_single_residual(self):
        """Single residual: both quantiles should be that value."""
        residuals = np.array([3.14])
        q_lo, q_hi = _corrected_residual_quantiles(residuals, 0.1)
        assert q_lo == 3.14
        assert q_hi == 3.14
