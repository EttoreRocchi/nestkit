"""Tests for multiclass OVR calibration, threshold optimization, and metrics in NestedCVClassifier."""

from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier

from nestkit import NestedCVClassifier


@pytest.fixture
def multiclass_clf():
    return RandomForestClassifier(n_estimators=10, random_state=42)


@pytest.fixture
def small_param_grid():
    return {"n_estimators": [10, 20]}


@pytest.mark.slow
class TestMulticlassWithCalibration:
    def test_ovr_calibration_sigmoid(self, multiclass_data, small_param_grid, multiclass_clf):
        """Verify OVR calibration fits per-class calibrators and renormalizes probabilities."""
        X, y = multiclass_data
        ncv = NestedCVClassifier(
            estimator=multiclass_clf,
            param_grid=small_param_grid,
            outer_cv=2,
            inner_cv=2,
            calibration_method="sigmoid",
            random_state=42,
        )
        ncv.fit(X, y)
        assert ncv.results_.has_calibration is True
        # Check calibrated probabilities exist in predictions
        pred_cols = ncv.results_.predictions_.columns
        assert any("y_proba_cal" in c for c in pred_cols)

    def test_ovr_calibration_isotonic(self, multiclass_data, small_param_grid, multiclass_clf):
        """Verify isotonic OVR calibration works for multiclass."""
        X, y = multiclass_data
        ncv = NestedCVClassifier(
            estimator=multiclass_clf,
            param_grid=small_param_grid,
            outer_cv=2,
            inner_cv=2,
            calibration_method="isotonic",
            random_state=42,
        )
        ncv.fit(X, y)
        assert ncv.results_.has_calibration is True


@pytest.mark.slow
class TestMulticlassWithThreshold:
    def test_threshold_pooled(self, multiclass_data, small_param_grid, multiclass_clf):
        """Verify OVR pooled threshold optimization for multiclass."""
        X, y = multiclass_data
        ncv = NestedCVClassifier(
            estimator=multiclass_clf,
            param_grid=small_param_grid,
            outer_cv=2,
            inner_cv=2,
            threshold_strategy="pooled",
            random_state=42,
        )
        ncv.fit(X, y)
        # OVR thresholds should be produced (no per-fold ThresholdResult for multiclass)
        assert ncv.results_.has_threshold_optimization is False  # multiclass uses OVR thresholds
        # Check optimized predictions exist
        pred_cols = ncv.results_.predictions_.columns
        assert "y_pred_optimized" in pred_cols

    def test_threshold_fold_specific(self, multiclass_data, small_param_grid, multiclass_clf):
        """Verify OVR fold-specific threshold optimization for multiclass."""
        X, y = multiclass_data
        ncv = NestedCVClassifier(
            estimator=multiclass_clf,
            param_grid=small_param_grid,
            outer_cv=2,
            inner_cv=2,
            threshold_strategy="fold_specific",
            random_state=42,
        )
        ncv.fit(X, y)
        pred_cols = ncv.results_.predictions_.columns
        assert "y_pred_optimized" in pred_cols


@pytest.mark.slow
class TestMulticlassWithCalibrationAndThreshold:
    def test_combined(self, multiclass_data, small_param_grid, multiclass_clf):
        """Verify combined OVR calibration and threshold optimization."""
        X, y = multiclass_data
        ncv = NestedCVClassifier(
            estimator=multiclass_clf,
            param_grid=small_param_grid,
            outer_cv=2,
            inner_cv=2,
            calibration_method="sigmoid",
            threshold_strategy="pooled",
            random_state=42,
        )
        ncv.fit(X, y)
        assert ncv.results_.has_calibration is True
        pred_cols = ncv.results_.predictions_.columns
        assert "y_pred_optimized" in pred_cols


@pytest.mark.slow
class TestMulticlassMetrics:
    def test_macro_metrics_present(self, multiclass_data, small_param_grid, multiclass_clf):
        """Verify macro-averaged precision, recall, f1, and multiclass roc_auc are computed."""
        X, y = multiclass_data
        ncv = NestedCVClassifier(
            estimator=multiclass_clf,
            param_grid=small_param_grid,
            outer_cv=2,
            inner_cv=2,
            random_state=42,
        )
        ncv.fit(X, y)
        scores_df = ncv.results_.outer_scores_default_
        for metric in ["precision", "recall", "f1", "roc_auc"]:
            assert metric in scores_df.columns, f"Missing metric: {metric}"


@pytest.mark.slow
class TestCustomCallableCriterion:
    def test_callable_criterion(self, binary_data, small_param_grid):
        """Verify custom callable threshold criterion is accepted."""
        from sklearn.metrics import f1_score

        def my_criterion(y_true, y_proba, threshold):
            y_pred = (y_proba >= threshold).astype(int)
            return f1_score(y_true, y_pred, zero_division=0.0)

        X, y = binary_data
        ncv = NestedCVClassifier(
            estimator=RandomForestClassifier(n_estimators=10, random_state=42),
            param_grid=small_param_grid,
            outer_cv=2,
            inner_cv=2,
            threshold_strategy="pooled",
            threshold_criterion=my_criterion,
            random_state=42,
        )
        ncv.fit(X, y)
        assert ncv.results_.has_threshold_optimization is True
