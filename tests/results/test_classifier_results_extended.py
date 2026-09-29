"""Extended tests for ClassifierResults covering reports, empty folds, to_json, and to_latex."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nestkit.conformal.results import ClassifierConformalResult
from nestkit.results.classifier_results import ClassifierOuterFoldResult, ClassifierResults
from nestkit.thresholding.results import ThresholdResult


def _make_fold_result(
    fold_idx=0,
    n_samples=20,
    with_calibration=False,
    with_threshold=False,
    with_conformal=False,
    rng=None,
):
    """Build a ClassifierOuterFoldResult for testing."""
    if rng is None:
        rng = np.random.RandomState(42 + fold_idx)

    y_true = rng.randint(0, 2, size=n_samples)
    y_proba = rng.rand(n_samples, 2)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    y_pred = (y_proba[:, 1] >= 0.5).astype(int)

    inner_cv_results = {
        "mean_test_score": np.array([0.85, 0.87]),
        "params": [{"n_estimators": 10}, {"n_estimators": 20}],
        "rank_test_score": np.array([2, 1]),
    }

    scores = {"accuracy": 0.85 + 0.02 * fold_idx, "f1": 0.83 + 0.01 * fold_idx}
    cm = np.array([[8, 2], [1, 9]])

    cal_proba = None
    cal_method = None
    cal_diag = None
    if with_calibration:
        cal_method = "sigmoid"
        cal_proba = y_proba * 0.95
        cal_diag = {
            "ece_raw": 0.10 + 0.01 * fold_idx,
            "ece_calibrated": 0.04 + 0.005 * fold_idx,
            "mce_raw": 0.15,
            "mce_calibrated": 0.08,
            "brier_raw": 0.20,
            "brier_calibrated": 0.15,
        }

    threshold_result = None
    y_pred_opt = None
    scores_opt = None
    cm_opt = None
    if with_threshold:
        threshold_result = ThresholdResult(
            strategy="pooled",
            optimal_threshold=0.42 + 0.02 * fold_idx,
            criterion_name="youden",
            criterion_value_at_optimum=0.7,
        )
        y_pred_opt = (y_proba[:, 1] >= threshold_result.optimal_threshold).astype(int)
        scores_opt = {"accuracy": 0.87 + 0.02 * fold_idx, "f1": 0.85 + 0.01 * fold_idx}
        cm_opt = np.array([[9, 1], [2, 8]])

    conformal_result = None
    conformal_sets = None
    conformal_sizes = None
    conformal_coverage = None
    if with_conformal:
        conformal_result = ClassifierConformalResult(
            alpha=0.1,
            qhat_per_class=np.array([0.3, 0.4]),
            n_calibration_per_class=np.array([80, 80]),
        )
        conformal_sets = [[0], [1], [0, 1]] * (n_samples // 3) + [[0]] * (n_samples % 3)
        conformal_sizes = np.array([len(s) for s in conformal_sets])
        conformal_coverage = float(
            np.mean([y_true[i] in conformal_sets[i] for i in range(n_samples)])
        )

    return ClassifierOuterFoldResult(
        fold_idx=fold_idx,
        train_indices=np.arange(160),
        test_indices=np.arange(n_samples) + fold_idx * n_samples,
        best_params={"n_estimators": 20},
        best_inner_score=0.87,
        inner_cv_results=inner_cv_results,
        fit_time=1.0,
        score_time=0.1,
        fitted_estimator=None,
        y_true=y_true,
        y_proba_raw=y_proba,
        y_pred_default=y_pred,
        outer_scores_default=scores,
        confusion_matrix_default=cm,
        y_proba_calibrated=cal_proba,
        calibration_method=cal_method,
        oof_calibration_diagnostics=cal_diag,
        y_pred_optimized=y_pred_opt,
        outer_scores_optimized=scores_opt,
        confusion_matrix_optimized=cm_opt,
        threshold_result=threshold_result,
        conformal_result=conformal_result,
        conformal_prediction_sets=conformal_sets,
        conformal_set_sizes=conformal_sizes,
        conformal_coverage=conformal_coverage,
    )


def _make_results(n_folds=3, **kwargs):
    """Build and finalize a ClassifierResults object."""
    results = ClassifierResults(n_outer_folds=n_folds)
    for i in range(n_folds):
        results.add_fold(_make_fold_result(fold_idx=i, **kwargs))
    results.finalize()
    return results


class TestEmptyFoldsProperties:
    def test_has_calibration_empty(self):
        """Verify has_calibration is False with no folds."""
        results = ClassifierResults(n_outer_folds=3)
        assert results.has_calibration is False

    def test_has_threshold_empty(self):
        """Verify has_threshold_optimization is False with no folds."""
        results = ClassifierResults(n_outer_folds=3)
        assert results.has_threshold_optimization is False

    def test_has_conformal_empty(self):
        """Verify has_conformal is False with no folds."""
        results = ClassifierResults(n_outer_folds=3)
        assert results.has_conformal is False


class TestThresholdComparison:
    def test_returns_dataframe(self):
        """Verify threshold_comparison returns a merged DataFrame."""
        results = _make_results(with_threshold=True)
        df = results.threshold_comparison()
        assert isinstance(df, pd.DataFrame)
        assert "mean_default" in df.columns
        assert "mean_optimized" in df.columns
        assert "metric" in df.columns

    def test_raises_without_threshold(self):
        """Verify ValueError when threshold optimization was not enabled."""
        results = _make_results(with_threshold=False)
        with pytest.raises(ValueError, match="not enabled"):
            results.threshold_comparison()


class TestCalibrationReport:
    def test_returns_dataframe(self):
        """Verify calibration_report returns a copy of calibration_summary_."""
        results = _make_results(with_calibration=True)
        df = results.calibration_report()
        assert isinstance(df, pd.DataFrame)
        assert "ece_raw" in df.columns

    def test_raises_without_calibration(self):
        """Verify ValueError when calibration was not enabled."""
        results = _make_results(with_calibration=False)
        with pytest.raises(ValueError, match="not enabled"):
            results.calibration_report()


class TestConformalReport:
    def test_returns_dataframe(self):
        """Verify conformal_report returns per-fold coverage and set sizes."""
        results = _make_results(with_conformal=True)
        df = results.conformal_report()
        assert isinstance(df, pd.DataFrame)
        assert "coverage" in df.columns
        assert "mean_set_size" in df.columns
        assert len(df) == 3

    def test_raises_without_conformal(self):
        """Verify ValueError when conformal prediction was not enabled."""
        results = _make_results(with_conformal=False)
        with pytest.raises(ValueError, match="not enabled"):
            results.conformal_report()


class TestClassificationReportPooled:
    def test_default(self):
        """Verify classification_report_pooled returns a string report."""
        results = _make_results()
        report = results.classification_report_pooled("default")
        assert isinstance(report, str)
        assert "precision" in report

    def test_optimized(self):
        """Verify classification_report_pooled works with optimized threshold."""
        results = _make_results(with_threshold=True)
        report = results.classification_report_pooled("optimized")
        assert isinstance(report, str)
        assert "precision" in report

    def test_optimized_raises_without_threshold(self):
        """Verify ValueError when requesting optimized report without threshold."""
        results = _make_results(with_threshold=False)
        with pytest.raises(ValueError, match="not enabled"):
            results.classification_report_pooled("optimized")


class TestConformalAttributes:
    def test_conformal_coverage(self):
        """Verify conformal_coverage_ contains mean and per_fold."""
        results = _make_results(with_conformal=True)
        assert "mean" in results.conformal_coverage_
        assert "per_fold" in results.conformal_coverage_
        assert len(results.conformal_coverage_["per_fold"]) == 3

    def test_conformal_set_size_stats(self):
        """Verify conformal_set_size_stats_ contains expected keys."""
        results = _make_results(with_conformal=True)
        stats = results.conformal_set_size_stats_
        for key in ["mean", "median", "frac_singleton", "frac_empty", "frac_multi"]:
            assert key in stats

    def test_conformal_qhat_stability(self):
        """Verify conformal_qhat_stability_ with multiple folds."""
        results = _make_results(n_folds=3, with_conformal=True)
        stab = results.conformal_qhat_stability_
        assert "mean_per_class" in stab
        assert "std_per_class" in stab


class TestToJsonAndLatex:
    def test_to_json_string(self):
        """Verify to_json returns valid JSON string."""
        results = _make_results()
        json_str = results.to_json()
        assert isinstance(json_str, str)
        import json

        data = json.loads(json_str)
        assert "n_outer_folds" in data

    def test_to_json_with_file(self, tmp_path):
        """Verify to_json writes to file when path is given."""
        results = _make_results()
        path = str(tmp_path / "results.json")
        json_str = results.to_json(path=path)
        assert isinstance(json_str, str)
        with open(path) as f:
            content = f.read()
        assert content == json_str

    def test_to_latex(self):
        """Verify to_latex returns a LaTeX tabular string."""
        results = _make_results()
        latex = results.to_latex()
        assert isinstance(latex, str)
        assert "tabular" in latex

    def test_to_json_with_numpy_types(self):
        """Verify to_json handles numpy int/float/array/DataFrame in _convert."""
        import json

        results = _make_results()
        # Inject numpy types that trigger json.dumps default handler
        results.best_params_per_fold_ = [
            {
                "n_estimators": np.int64(20),
                "weights": np.array([0.1, 0.2, 0.3]),
                "summary": pd.DataFrame({"a": [1, 2]}),
            },
        ]
        json_str = results.to_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["best_params_per_fold"][0]["n_estimators"] == 20
        assert data["best_params_per_fold"][0]["weights"] == [0.1, 0.2, 0.3]
        assert isinstance(data["best_params_per_fold"][0]["summary"], list)

    def test_to_latex_without_summary(self):
        """Verify to_latex returns empty string when summary_default_ is absent."""
        results = ClassifierResults(n_outer_folds=3)
        latex = results.to_latex()
        assert latex == ""

    def test_to_dataframe(self):
        """Verify to_dataframe returns a copy of outer_scores_default_."""
        results = _make_results()
        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_to_dataframe_without_finalize(self):
        """Verify to_dataframe returns empty DataFrame before finalize."""
        results = ClassifierResults(n_outer_folds=3)
        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestCalibrationImprovement:
    def test_calibration_improvement_computed(self):
        """Verify calibration_improvement_ DataFrame is computed."""
        results = _make_results(with_calibration=True)
        df = results.calibration_improvement_
        assert isinstance(df, pd.DataFrame)
        assert "delta_ece" in df.columns
        assert "delta_brier" in df.columns


class TestThresholdStability:
    def test_threshold_stability_dict(self):
        """Verify threshold_stability_ contains mean, std, cv, range."""
        results = _make_results(with_threshold=True)
        stab = results.threshold_stability_
        for key in ["mean", "std", "cv", "range"]:
            assert key in stab
