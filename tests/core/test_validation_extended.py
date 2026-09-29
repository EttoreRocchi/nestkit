"""Extended validation tests covering Mondrian params and survival target warnings."""

from __future__ import annotations

import numpy as np
import pytest

from nestkit._validation import validate_mondrian_params, validate_survival_target


class TestValidateMondrianParams:
    def test_bins_non_int_float(self):
        """Verify ValueError when mondrian_bins is a float."""
        with pytest.raises(ValueError, match="positive integer"):
            validate_mondrian_params(mondrian_bins=2.5, mondrian_min_bin_size=10)

    def test_bins_zero(self):
        """Verify ValueError when mondrian_bins is zero."""
        with pytest.raises(ValueError, match="positive integer"):
            validate_mondrian_params(mondrian_bins=0, mondrian_min_bin_size=10)

    def test_bins_negative(self):
        """Verify ValueError when mondrian_bins is negative."""
        with pytest.raises(ValueError, match="positive integer"):
            validate_mondrian_params(mondrian_bins=-1, mondrian_min_bin_size=10)

    def test_min_bin_size_non_int(self):
        """Verify ValueError when mondrian_min_bin_size is a float."""
        with pytest.raises(ValueError, match="positive integer"):
            validate_mondrian_params(mondrian_bins=5, mondrian_min_bin_size=10.5)

    def test_min_bin_size_zero(self):
        """Verify ValueError when mondrian_min_bin_size is zero."""
        with pytest.raises(ValueError, match="positive integer"):
            validate_mondrian_params(mondrian_bins=5, mondrian_min_bin_size=0)

    def test_valid_params_accepted(self):
        """Verify valid params don't raise."""
        validate_mondrian_params(mondrian_bins=5, mondrian_min_bin_size=10)

    def test_bins_none_accepted(self):
        """Verify mondrian_bins=None is accepted (disables Mondrian)."""
        validate_mondrian_params(mondrian_bins=None, mondrian_min_bin_size=10)


class TestValidateSurvivalTargetWarnings:
    def test_high_censoring_rate_warns(self):
        """Verify UserWarning when censoring rate > 95%."""
        n = 200
        event = np.zeros(n)
        event[:5] = 1.0  # 2.5% event rate -> 97.5% censoring
        duration = np.random.RandomState(42).exponential(10, n)
        y = np.column_stack([event, duration])
        with pytest.warns(UserWarning, match="Very high censoring rate"):
            validate_survival_target(y)

    def test_low_censoring_rate_warns(self):
        """Verify UserWarning when censoring rate < 5%."""
        n = 200
        event = np.ones(n)
        event[:5] = 0.0  # 2.5% censoring
        duration = np.random.RandomState(42).exponential(10, n)
        y = np.column_stack([event, duration])
        with pytest.warns(UserWarning, match="Very low censoring rate"):
            validate_survival_target(y)

    def test_normal_censoring_no_warning(self):
        """Verify no warning for normal censoring rate (~50%)."""
        import warnings

        n = 200
        rng = np.random.RandomState(42)
        event = rng.randint(0, 2, n).astype(float)
        duration = rng.exponential(10, n)
        y = np.column_stack([event, duration])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_survival_target(y)  # Should not raise any warning

    def test_wrong_shape_raises(self):
        """Verify ValueError for wrong target shape."""
        y = np.ones((10, 3))
        with pytest.raises(ValueError, match="shape"):
            validate_survival_target(y)

    def test_non_binary_event_raises(self):
        """Verify ValueError for non-binary event column."""
        y = np.column_stack([np.array([0, 1, 2, 0, 1]), np.ones(5)])
        with pytest.raises(ValueError, match="binary"):
            validate_survival_target(y)

    def test_non_positive_duration_raises(self):
        """Verify ValueError for non-positive duration."""
        y = np.column_stack([np.array([0, 1, 0, 1, 0]), np.array([1, 2, -1, 3, 4])])
        with pytest.raises(ValueError, match="positive"):
            validate_survival_target(y)
