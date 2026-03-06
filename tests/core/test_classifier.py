"""End-to-end tests for NestedCVClassifier."""

import pandas as pd

from nestkit import NestedCVClassifier


class TestBasicBinary:
    """Test basic binary classification with default settings."""

    def test_basic_binary(self, binary_data, simple_param_grid, simple_classifier):
        X, y = binary_data
        ncv = NestedCVClassifier(
            estimator=simple_classifier,
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
        assert "y_pred_default" in results.predictions_.columns

        assert hasattr(results, "outer_scores_default_")
        assert len(results.outer_scores_default_) == 3

        assert len(results.best_params_per_fold_) == 3

        assert results.has_calibration is False
        assert results.has_threshold_optimization is False


class TestBasicMulticlass:
    """Test basic multiclass classification."""

    def test_basic_multiclass(self, multiclass_data, simple_param_grid, simple_classifier):
        X, y = multiclass_data
        ncv = NestedCVClassifier(
            estimator=simple_classifier,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
        )
        ncv.fit(X, y)

        assert ncv.is_fitted_ is True
        results = ncv.results_
        assert isinstance(results.summary_default_, pd.DataFrame)
        assert isinstance(results.predictions_, pd.DataFrame)
        assert len(results.outer_scores_default_) == 3
        assert results.has_calibration is False
        assert results.has_threshold_optimization is False


class TestWithCalibration:
    """Test with post-hoc calibration enabled."""

    def test_with_calibration(self, binary_data, simple_param_grid, simple_classifier):
        X, y = binary_data
        ncv = NestedCVClassifier(
            estimator=simple_classifier,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
            calibration_method="sigmoid",
        )
        ncv.fit(X, y)

        results = ncv.results_
        assert results.has_calibration is True
        assert results.has_threshold_optimization is False

        assert hasattr(results, "calibration_summary_")
        assert isinstance(results.calibration_summary_, pd.DataFrame)

        cal_cols = [c for c in results.predictions_.columns if "cal" in c]
        assert len(cal_cols) > 0


class TestWithThresholdPooled:
    """Test with pooled threshold optimization."""

    def test_with_threshold_pooled(self, binary_data, simple_param_grid, simple_classifier):
        X, y = binary_data
        ncv = NestedCVClassifier(
            estimator=simple_classifier,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
            threshold_strategy="pooled",
        )
        ncv.fit(X, y)

        results = ncv.results_
        assert results.has_threshold_optimization is True

        assert hasattr(results, "summary_optimized_")
        assert isinstance(results.summary_optimized_, pd.DataFrame)
        assert hasattr(results, "thresholds_per_fold_")
        assert len(results.thresholds_per_fold_) == 3
        assert hasattr(results, "threshold_stability_")
        assert "mean" in results.threshold_stability_

        assert "y_pred_optimized" in results.predictions_.columns


class TestWithCalibrationAndThreshold:
    """Test with both calibration and threshold optimization."""

    def test_with_calibration_and_threshold(
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
        )
        ncv.fit(X, y)

        results = ncv.results_
        assert results.has_calibration is True
        assert results.has_threshold_optimization is True

        cal_cols = [c for c in results.predictions_.columns if "cal" in c]
        assert len(cal_cols) > 0
        assert "y_pred_optimized" in results.predictions_.columns


class TestDataFrameInput:
    """Test with pandas DataFrame input."""

    def test_dataframe_input(self, binary_data_df, simple_param_grid, simple_classifier):
        X_df, y_series = binary_data_df
        ncv = NestedCVClassifier(
            estimator=simple_classifier,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
        )
        ncv.fit(X_df, y_series)

        assert ncv.feature_names_in_ == [f"feat_{i}" for i in range(10)]

        results = ncv.results_
        original_index = X_df.index
        for idx in results.predictions_.index:
            assert idx in original_index


class TestScoringDict:
    """Test with a dictionary of scoring metrics."""

    def test_scoring_dict(self, binary_data, simple_param_grid, simple_classifier):
        X, y = binary_data
        ncv = NestedCVClassifier(
            estimator=simple_classifier,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
            scoring={"acc": "accuracy", "f1": "f1"},
        )
        ncv.fit(X, y)

        assert ncv.is_fitted_ is True
        results = ncv.results_
        assert isinstance(results.summary_default_, pd.DataFrame)
        assert len(results.outer_scores_default_) == 3
