"""End-to-end tests for NestedCVRegressor."""

import numpy as np
import pandas as pd

from nestkit import NestedCVRegressor


class TestBasicRegression:
    """Test basic regression with default settings."""

    def test_basic(self, regression_data, simple_param_grid, simple_regressor):
        X, y = regression_data
        ncv = NestedCVRegressor(
            estimator=simple_regressor,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
        )
        ncv.fit(X, y)

        assert ncv.is_fitted_ is True

        results = ncv.results_
        assert results is not None

        assert hasattr(results, "summary_default_")
        assert isinstance(results.summary_default_, pd.DataFrame)
        assert "metric" in results.summary_default_.columns
        assert "mean" in results.summary_default_.columns

        assert hasattr(results, "predictions_")
        assert isinstance(results.predictions_, pd.DataFrame)
        assert "y_true" in results.predictions_.columns
        assert "y_pred" in results.predictions_.columns
        assert "residual" in results.predictions_.columns

        assert hasattr(results, "outer_scores_default_")
        assert len(results.outer_scores_default_) == 3

        assert len(results.best_params_per_fold_) == 3

        assert hasattr(results, "residual_stats_")
        assert "mean" in results.residual_stats_
        assert "std" in results.residual_stats_

        assert results.prediction_interval_coverage_ is None


class TestPredictionIntervals:
    """Test with prediction intervals enabled."""

    def test_prediction_intervals(self, regression_data, simple_param_grid, simple_regressor):
        X, y = regression_data
        ncv = NestedCVRegressor(
            estimator=simple_regressor,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
            prediction_intervals=True,
            confidence_level=0.95,
        )
        ncv.fit(X, y)

        results = ncv.results_

        assert results.prediction_interval_coverage_ is not None
        assert "mean" in results.prediction_interval_coverage_
        assert "per_fold" in results.prediction_interval_coverage_
        assert len(results.prediction_interval_coverage_["per_fold"]) == 3

        for cov in results.prediction_interval_coverage_["per_fold"]:
            assert 0.0 <= cov <= 1.0

        assert "pi_lower" in results.predictions_.columns
        assert "pi_upper" in results.predictions_.columns

        assert (results.predictions_["pi_lower"] <= results.predictions_["pi_upper"]).all()


class TestMetrics:
    """Test that expected regression metrics are present in outer scores."""

    def test_metrics(self, regression_data, simple_param_grid, simple_regressor):
        X, y = regression_data
        ncv = NestedCVRegressor(
            estimator=simple_regressor,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
        )
        ncv.fit(X, y)

        results = ncv.results_
        expected_metrics = {"mse", "rmse", "mae", "r2"}
        outer_cols = set(results.outer_scores_default_.columns)
        assert expected_metrics.issubset(outer_cols), (
            f"Missing metrics: {expected_metrics - outer_cols}"
        )

        for _, row in results.outer_scores_default_.iterrows():
            assert row["mse"] >= 0
            assert row["rmse"] >= 0
            assert row["mae"] >= 0
            np.testing.assert_almost_equal(row["rmse"], np.sqrt(row["mse"]), decimal=5)
