"""Tests for nestkit.results._base -- _BaseNestedCVResults.

Uses RegressorResults as the concrete subclass for testing.
"""

import json

import numpy as np

from nestkit.results.regressor_results import RegressorOuterFoldResult, RegressorResults


def _make_fold(fold_idx, n_samples=30, seed=None):
    """Create a minimal RegressorOuterFoldResult."""
    rng = np.random.RandomState(seed if seed is not None else fold_idx)
    y_true = rng.normal(0, 1, size=n_samples)
    y_pred = y_true + rng.normal(0, 0.1, size=n_samples)
    return RegressorOuterFoldResult(
        fold_idx=fold_idx,
        train_indices=np.arange(200),
        test_indices=np.arange(fold_idx * n_samples, (fold_idx + 1) * n_samples),
        best_params={"alpha": 1.0},
        best_inner_score=0.85,
        inner_cv_results={
            "mean_test_score": [0.80, 0.85],
            "rank_test_score": [2, 1],
            "param_alpha": [0.1, 1.0],
        },
        fit_time=0.1,
        score_time=0.01,
        fitted_estimator=None,
        y_true=y_true,
        y_pred=y_pred,
        outer_scores={"r2": 0.9 + fold_idx * 0.02},
        residuals=y_true - y_pred,
    )


def _finalized_results(n_folds=3, feature_names=None):
    """Build and finalize a RegressorResults."""
    results = RegressorResults(n_outer_folds=n_folds, feature_names=feature_names)
    for i in range(n_folds):
        results.add_fold(_make_fold(i))
    results.finalize()
    return results


class TestBaseNestedCVResults:
    def test_to_dict_keys(self):
        results = _finalized_results()
        d = results.to_dict()
        assert "n_outer_folds" in d
        assert "feature_names" in d
        assert d["n_outer_folds"] == 3

    def test_to_dict_includes_best_params_after_finalize(self):
        results = _finalized_results()
        d = results.to_dict()
        assert "best_params_per_fold" in d

    def test_to_dict_no_best_params_before_finalize(self):
        results = RegressorResults(n_outer_folds=2)
        d = results.to_dict()
        assert "best_params_per_fold" not in d

    def test_to_dataframe_returns_copy(self):
        results = _finalized_results()
        df = results.to_dataframe()
        df["r2"] = 0.0
        assert results.outer_scores_default_["r2"].iloc[0] != 0.0

    def test_to_dataframe_before_finalize_returns_empty(self):
        results = RegressorResults(n_outer_folds=2)
        df = results.to_dataframe()
        assert len(df) == 0

    def test_to_dataframe_correct_shape(self):
        results = _finalized_results(n_folds=3)
        df = results.to_dataframe()
        assert len(df) == 3
        assert "r2" in df.columns

    def test_to_json_returns_valid_json_string(self):
        results = _finalized_results()
        json_str = results.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["n_outer_folds"] == 3

    def test_to_json_with_path_writes_file(self, tmp_path):
        results = _finalized_results()
        path = str(tmp_path / "results.json")
        results.to_json(path)
        with open(path) as f:
            parsed = json.loads(f.read())
        assert parsed["n_outer_folds"] == 3

    def test_to_json_handles_numpy_types(self):
        results = _finalized_results()
        json_str = results.to_json()
        json.loads(json_str)

    def test_to_latex_returns_string_with_tabular(self):
        results = _finalized_results()
        latex = results.to_latex()
        assert "tabular" in latex

    def test_to_latex_before_finalize_returns_empty(self):
        results = RegressorResults(n_outer_folds=2)
        assert results.to_latex() == ""

    def test_compute_summary_columns(self):
        results = _finalized_results()
        summary = results.summary_default_
        for col in ("metric", "mean", "std", "ci_lower", "ci_upper", "median", "iqr"):
            assert col in summary.columns

    def test_compute_summary_ci_with_single_fold(self):
        results = _finalized_results(n_folds=1)
        summary = results.summary_default_
        row = summary.iloc[0]
        assert row["ci_lower"] == row["ci_upper"] == row["mean"]

    def test_compute_summary_ci_bounds(self):
        results = _finalized_results(n_folds=3)
        summary = results.summary_default_
        for _, row in summary.iterrows():
            assert row["ci_lower"] <= row["mean"] <= row["ci_upper"]

    def test_has_fitted_estimators_false_when_no_folds(self):
        results = RegressorResults(n_outer_folds=2)
        assert results.has_fitted_estimators is False

    def test_has_fitted_estimators_false_when_none(self):
        results = _finalized_results()
        assert results.has_fitted_estimators is False

    def test_add_fold_increments_list(self):
        results = RegressorResults(n_outer_folds=3)
        assert len(results.fold_results_) == 0
        results.add_fold(_make_fold(0))
        assert len(results.fold_results_) == 1
        results.add_fold(_make_fold(1))
        assert len(results.fold_results_) == 2

    def test_feature_names_preserved(self):
        names = ["a", "b", "c"]
        results = _finalized_results(feature_names=names)
        assert results.feature_names_in_ == names
        d = results.to_dict()
        assert d["feature_names"] == names
