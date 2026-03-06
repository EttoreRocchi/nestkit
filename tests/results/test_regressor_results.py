"""Tests for nestkit.results.regressor_results.

Covers: RegressorOuterFoldResult, RegressorResults, _skewness, _kurtosis.
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nestkit.results.regressor_results import (
    RegressorOuterFoldResult,
    RegressorResults,
    _kurtosis,
    _skewness,
)


def _make_regressor_fold_result(fold_idx, n_samples=30, with_pi=False, seed=None):
    """Create a minimal mock RegressorOuterFoldResult."""
    rng = np.random.RandomState(seed if seed is not None else fold_idx)
    y_true = rng.normal(0, 1, size=n_samples)
    y_pred = y_true + rng.normal(0, 0.1, size=n_samples)
    residuals = y_true - y_pred
    test_indices = np.arange(fold_idx * n_samples, (fold_idx + 1) * n_samples)
    train_indices = np.arange(200)

    kwargs = dict(
        fold_idx=fold_idx,
        train_indices=train_indices,
        test_indices=test_indices,
        best_params={"alpha": 1.0},
        best_inner_score=0.85,
        inner_cv_results={
            "mean_test_score": [0.80, 0.85, 0.90],
            "std_test_score": [0.02, 0.01, 0.03],
            "rank_test_score": [3, 2, 1],
            "param_alpha": [0.01, 0.1, 1.0],
        },
        fit_time=0.1,
        score_time=0.01,
        fitted_estimator=None,
        y_true=y_true,
        y_pred=y_pred,
        outer_scores={"r2": 0.9 + fold_idx * 0.01, "neg_mse": -(0.1 - fold_idx * 0.01)},
        residuals=residuals,
    )
    if with_pi:
        kwargs["prediction_interval_lower"] = y_pred - 0.5
        kwargs["prediction_interval_upper"] = y_pred + 0.5
        kwargs["coverage"] = 0.95

    return RegressorOuterFoldResult(**kwargs)


class TestSkewness:
    def test_normal_distribution_near_zero(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, size=10000)
        assert abs(_skewness(x)) < 0.1

    def test_positive_skew(self):
        rng = np.random.RandomState(42)
        x = rng.exponential(1, size=1000)
        assert _skewness(x) > 0.5

    def test_fewer_than_3_returns_zero(self):
        assert _skewness(np.array([1.0, 2.0])) == 0.0
        assert _skewness(np.array([1.0])) == 0.0

    def test_constant_values_returns_zero(self):
        assert _skewness(np.array([5.0, 5.0, 5.0, 5.0])) == 0.0

    @given(
        arrays(
            float,
            shape=st.integers(3, 100),
            elements=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=50)
    def test_skewness_is_finite(self, arr):
        result = _skewness(arr)
        assert np.isfinite(result)


class TestKurtosis:
    def test_normal_distribution_near_zero(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, size=10000)
        assert abs(_kurtosis(x)) < 0.2

    def test_fewer_than_4_returns_zero(self):
        assert _kurtosis(np.array([1.0, 2.0, 3.0])) == 0.0
        assert _kurtosis(np.array([1.0])) == 0.0

    def test_constant_values_returns_zero(self):
        assert _kurtosis(np.array([5.0, 5.0, 5.0, 5.0, 5.0])) == 0.0

    def test_uniform_distribution_negative_kurtosis(self):
        rng = np.random.RandomState(42)
        x = rng.uniform(0, 1, size=10000)
        assert _kurtosis(x) < 0

    @given(
        arrays(
            float,
            shape=st.integers(4, 100),
            elements=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=50)
    def test_kurtosis_is_finite(self, arr):
        result = _kurtosis(arr)
        assert np.isfinite(result)


class TestRegressorOuterFoldResult:
    def test_dataclass_fields_stored(self):
        fr = _make_regressor_fold_result(0)
        assert fr.fold_idx == 0
        assert fr.best_params == {"alpha": 1.0}
        assert len(fr.y_true) == 30
        assert len(fr.y_pred) == 30

    def test_default_prediction_interval_none(self):
        fr = _make_regressor_fold_result(0)
        assert fr.prediction_interval_lower is None
        assert fr.prediction_interval_upper is None
        assert fr.coverage is None


class TestRegressorResults:
    def test_finalize_computes_summary(self):
        results = RegressorResults(n_outer_folds=3)
        for i in range(3):
            results.add_fold(_make_regressor_fold_result(i))
        results.finalize()
        assert hasattr(results, "summary_default_")
        assert hasattr(results, "best_params_per_fold_")
        assert hasattr(results, "inner_reports_")
        assert len(results.summary_default_) > 0

    def test_finalize_idempotent(self):
        results = RegressorResults(n_outer_folds=2)
        for i in range(2):
            results.add_fold(_make_regressor_fold_result(i))
        results.finalize()
        summary1 = results.summary_default_.copy()
        results.finalize()
        pd.testing.assert_frame_equal(results.summary_default_, summary1)

    def test_predictions_dataframe_structure(self):
        results = RegressorResults(n_outer_folds=3)
        for i in range(3):
            results.add_fold(_make_regressor_fold_result(i))
        results.finalize()
        preds = results.predictions_
        assert "y_true" in preds.columns
        assert "y_pred" in preds.columns
        assert "residual" in preds.columns
        assert "fold_idx" in preds.columns

    def test_predictions_with_prediction_intervals(self):
        results = RegressorResults(n_outer_folds=2)
        for i in range(2):
            results.add_fold(_make_regressor_fold_result(i, with_pi=True))
        results.finalize()
        preds = results.predictions_
        assert "pi_lower" in preds.columns
        assert "pi_upper" in preds.columns

    def test_generalization_gap_computed(self):
        results = RegressorResults(n_outer_folds=3)
        for i in range(3):
            results.add_fold(_make_regressor_fold_result(i))
        results.finalize()
        gap = results.generalization_gap_
        assert isinstance(gap, pd.DataFrame)
        assert "fold_idx" in gap.columns
        assert "best_inner_score" in gap.columns

    def test_residual_stats_keys(self):
        results = RegressorResults(n_outer_folds=3)
        for i in range(3):
            results.add_fold(_make_regressor_fold_result(i))
        results.finalize()
        stats = results.residual_stats_
        for key in ("mean", "std", "median", "skewness", "kurtosis"):
            assert key in stats

    def test_prediction_interval_coverage_none_when_no_pi(self):
        results = RegressorResults(n_outer_folds=2)
        for i in range(2):
            results.add_fold(_make_regressor_fold_result(i, with_pi=False))
        results.finalize()
        assert results.prediction_interval_coverage_ is None

    def test_prediction_interval_coverage_computed(self):
        results = RegressorResults(n_outer_folds=2)
        for i in range(2):
            results.add_fold(_make_regressor_fold_result(i, with_pi=True))
        results.finalize()
        cov = results.prediction_interval_coverage_
        assert cov is not None
        assert "mean" in cov
        assert "per_fold" in cov

    def test_original_index_preserved(self):
        idx = pd.RangeIndex(0, 60)
        results = RegressorResults(n_outer_folds=2, original_index=idx)
        for i in range(2):
            results.add_fold(_make_regressor_fold_result(i))
        results.finalize()
        assert len(results.predictions_) == 60
