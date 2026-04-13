"""Integration tests for conformal prediction in NestedCVClassifier/Regressor."""

import pandas as pd
import pytest

from nestkit import NestedCVClassifier, NestedCVRegressor

pytestmark = pytest.mark.slow


class TestClassifierConformalBinary:
    """Binary classifier with conformal prediction only."""

    def test_conformal_only(self, binary_data, simple_param_grid, simple_classifier):
        X, y = binary_data
        ncv = NestedCVClassifier(
            estimator=simple_classifier,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
            conformal_prediction=True,
            conformal_alpha=0.1,
        )
        ncv.fit(X, y)

        results = ncv.results_
        assert results.has_conformal is True
        assert results.has_calibration is False

        # Check conformal attributes exist
        assert hasattr(results, "conformal_coverage_")
        assert hasattr(results, "conformal_set_size_stats_")
        assert hasattr(results, "conformal_qhat_per_fold_")
        assert hasattr(results, "conformal_qhat_stability_")

        # Coverage should be reasonable
        assert 0 < results.conformal_coverage_["mean"] <= 1.0

        # Predictions DataFrame should have conformal columns
        assert "conformal_set_size" in results.predictions_.columns
        assert "conformal_in_set" in results.predictions_.columns

        # conformal_report should work
        report = results.conformal_report()
        assert isinstance(report, pd.DataFrame)
        assert len(report) == 3  # 3 outer folds
        assert "coverage" in report.columns

    def test_conformal_with_calibration(self, binary_data, simple_param_grid, simple_classifier):
        X, y = binary_data
        ncv = NestedCVClassifier(
            estimator=simple_classifier,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
            calibration_method="sigmoid",
            conformal_prediction=True,
            conformal_alpha=0.1,
        )
        ncv.fit(X, y)

        results = ncv.results_
        assert results.has_conformal is True
        assert results.has_calibration is True

    def test_conformal_with_calibration_and_threshold(
        self, binary_data, simple_param_grid, simple_classifier
    ):
        X, y = binary_data
        ncv = NestedCVClassifier(
            estimator=simple_classifier,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
            calibration_method="sigmoid",
            threshold_strategy="pooled",
            conformal_prediction=True,
            conformal_alpha=0.1,
        )
        ncv.fit(X, y)

        results = ncv.results_
        assert results.has_conformal is True
        assert results.has_calibration is True
        assert results.has_threshold_optimization is True


class TestClassifierConformalMulticlass:
    """Multiclass classifier with conformal prediction."""

    def test_multiclass_conformal(self, multiclass_data, simple_param_grid, simple_classifier):
        X, y = multiclass_data
        ncv = NestedCVClassifier(
            estimator=simple_classifier,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
            conformal_prediction=True,
            conformal_alpha=0.1,
        )
        ncv.fit(X, y)

        results = ncv.results_
        assert results.has_conformal is True

        # q_hat should have 3 entries per fold
        assert results.conformal_qhat_per_fold_.shape[1] == 3

        # Set sizes can range 0-3
        max_size = results.predictions_["conformal_set_size"].max()
        assert max_size <= 3


class TestClassifierConformalValidation:
    """Parameter validation for conformal prediction."""

    def test_invalid_alpha_raises(self, binary_data, simple_param_grid, simple_classifier):
        X, y = binary_data
        ncv = NestedCVClassifier(
            estimator=simple_classifier,
            param_grid=simple_param_grid,
            conformal_prediction=True,
            conformal_alpha=1.5,
        )
        with pytest.raises(ValueError, match="conformal_alpha"):
            ncv.fit(X, y)

    def test_conformal_report_raises_when_disabled(
        self, binary_data, simple_param_grid, simple_classifier
    ):
        X, y = binary_data
        ncv = NestedCVClassifier(
            estimator=simple_classifier,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
        )
        ncv.fit(X, y)
        with pytest.raises(ValueError, match="not enabled"):
            ncv.results_.conformal_report()


class TestRegressorMondrian:
    """Regressor with Mondrian prediction intervals."""

    def test_mondrian_intervals(self, regression_data, simple_param_grid, simple_regressor):
        X, y = regression_data
        ncv = NestedCVRegressor(
            estimator=simple_regressor,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
            prediction_intervals=True,
            mondrian_bins=3,
        )
        ncv.fit(X, y)

        results = ncv.results_
        assert results.prediction_interval_coverage_ is not None
        assert results.mondrian_coverage_per_bin_ is not None
        assert len(results.mondrian_coverage_per_bin_) > 0

        # Predictions should have bin assignments
        assert "mondrian_bin" in results.predictions_.columns
        assert "pi_lower" in results.predictions_.columns
        assert "pi_upper" in results.predictions_.columns

    def test_mondrian_without_prediction_intervals(
        self, regression_data, simple_param_grid, simple_regressor
    ):
        """mondrian_bins has no effect if prediction_intervals=False."""
        X, y = regression_data
        ncv = NestedCVRegressor(
            estimator=simple_regressor,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
            prediction_intervals=False,
            mondrian_bins=3,
        )
        ncv.fit(X, y)

        results = ncv.results_
        assert results.prediction_interval_coverage_ is None
        assert results.mondrian_coverage_per_bin_ is None

    def test_global_intervals_still_work(
        self, regression_data, simple_param_grid, simple_regressor
    ):
        """Ensure global (non-Mondrian) intervals still work as before."""
        X, y = regression_data
        ncv = NestedCVRegressor(
            estimator=simple_regressor,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
            prediction_intervals=True,
        )
        ncv.fit(X, y)

        results = ncv.results_
        assert results.prediction_interval_coverage_ is not None
        assert results.mondrian_coverage_per_bin_ is None
        assert "pi_lower" in results.predictions_.columns

    def test_invalid_mondrian_bins_raises(
        self, regression_data, simple_param_grid, simple_regressor
    ):
        X, y = regression_data
        ncv = NestedCVRegressor(
            estimator=simple_regressor,
            param_grid=simple_param_grid,
            mondrian_bins=0,
        )
        with pytest.raises(ValueError, match="mondrian_bins"):
            ncv.fit(X, y)
