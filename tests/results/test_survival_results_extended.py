"""Extended tests for SurvivalResults covering coefficient stability edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nestkit.results.survival_results import SurvivalOuterFoldResult, SurvivalResults


def _make_survival_fold(fold_idx=0, n_samples=20, coefficients=None, rng=None):
    """Build a SurvivalOuterFoldResult for testing."""
    if rng is None:
        rng = np.random.RandomState(42 + fold_idx)

    event = rng.randint(0, 2, size=n_samples).astype(float)
    duration = rng.exponential(10, size=n_samples)
    risk_scores = rng.randn(n_samples)

    inner_cv_results = {
        "mean_test_score": np.array([0.65, 0.67]),
        "params": [{"penalizer": 0.0}, {"penalizer": 0.1}],
        "rank_test_score": np.array([2, 1]),
    }

    return SurvivalOuterFoldResult(
        fold_idx=fold_idx,
        train_indices=np.arange(160),
        test_indices=np.arange(n_samples) + fold_idx * n_samples,
        best_params={"penalizer": 0.1},
        best_inner_score=0.67,
        inner_cv_results=inner_cv_results,
        fit_time=1.0,
        score_time=0.1,
        fitted_estimator=None,
        y_event=event,
        y_duration=duration,
        risk_scores=risk_scores,
        outer_scores={"c_index": 0.65 + 0.02 * fold_idx, "uno_c_index": 0.63 + 0.02 * fold_idx},
        coefficients=coefficients,
    )


class TestCoefficientStabilityNoCoefficients:
    def test_empty_when_all_none(self):
        """Verify coefficient_stability_ is empty when all folds have no coefficients."""
        results = SurvivalResults(n_outer_folds=3, feature_names=["x0", "x1", "x2"])
        for i in range(3):
            results.add_fold(_make_survival_fold(fold_idx=i, coefficients=None))
        results.finalize()
        assert isinstance(results.coefficient_stability_, pd.DataFrame)
        assert results.coefficient_stability_.empty


class TestCoefficientStabilityWithCoefficients:
    def test_computed_correctly(self):
        """Verify coefficient_stability_ computes mean, std, and hazard ratios."""
        feature_names = ["x0", "x1", "x2"]
        results = SurvivalResults(n_outer_folds=3, feature_names=feature_names)
        for i in range(3):
            coefs = {"x0": 0.5 + 0.1 * i, "x1": -0.3 + 0.05 * i, "x2": 0.2}
            results.add_fold(_make_survival_fold(fold_idx=i, coefficients=coefs))
        results.finalize()

        df = results.coefficient_stability_
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "feature" in df.columns
        assert "coef_mean" in df.columns
        assert "coef_std" in df.columns
        assert "hazard_ratio_mean" in df.columns
        assert len(df) == 3


class TestSurvivalResultsFinalize:
    def test_predictions_dataframe(self):
        """Verify predictions_ DataFrame is built correctly."""
        results = SurvivalResults(n_outer_folds=2, feature_names=["x0", "x1"])
        for i in range(2):
            results.add_fold(_make_survival_fold(fold_idx=i))
        results.finalize()

        assert isinstance(results.predictions_, pd.DataFrame)
        assert "y_event" in results.predictions_.columns
        assert "y_duration" in results.predictions_.columns
        assert "risk_score" in results.predictions_.columns
        assert "fold_idx" in results.predictions_.columns

    def test_generalization_gap(self):
        """Verify generalization_gap_ is computed."""
        results = SurvivalResults(n_outer_folds=2, feature_names=["x0"])
        for i in range(2):
            results.add_fold(_make_survival_fold(fold_idx=i))
        results.finalize()

        assert isinstance(results.generalization_gap_, pd.DataFrame)
        assert "best_inner_score" in results.generalization_gap_.columns

    def test_summary_default(self):
        """Verify summary_default_ is computed with correct columns."""
        results = SurvivalResults(n_outer_folds=3, feature_names=["x0"])
        for i in range(3):
            results.add_fold(_make_survival_fold(fold_idx=i))
        results.finalize()

        assert isinstance(results.summary_default_, pd.DataFrame)
        assert "metric" in results.summary_default_.columns
        assert "mean" in results.summary_default_.columns
