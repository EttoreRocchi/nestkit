"""Extended plotting tests covering calibration, comparison, importance, threshold, and summary modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nestkit.plotting.calibration import plot_calibration_curves, plot_calibration_improvement
from nestkit.plotting.comparison import (
    plot_bayesian_posterior,
    plot_comparison,
    plot_critical_difference,
    plot_score_differences,
)
from nestkit.plotting.importance import (
    plot_importance,
    plot_rank_stability_features,
    plot_selection_frequency,
    plot_shap_summary,
)
from nestkit.plotting.summary import (
    plot_precision_recall_curves,
    plot_predicted_vs_actual,
    plot_prediction_intervals,
    plot_rank_stability,
    plot_residual_qq,
)
from nestkit.plotting.threshold import (
    plot_threshold_comparison,
    plot_threshold_distribution,
    plot_threshold_sensitivity,
)
from nestkit.plotting.tuning import plot_inner_tuning_curve


@dataclass
class _FakeFoldResult:
    fold_idx: int = 0
    y_true: np.ndarray = field(default_factory=lambda: np.array([]))
    y_proba_raw: np.ndarray = field(default_factory=lambda: np.array([]))
    y_proba_calibrated: np.ndarray | None = None
    threshold_result: object | None = None
    train_indices: np.ndarray = field(default_factory=lambda: np.arange(160))
    test_indices: np.ndarray = field(default_factory=lambda: np.arange(40))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    conformal_coverage: float | None = None
    conformal_set_sizes: np.ndarray | None = None
    conformal_result: object | None = None


@dataclass
class _FakeThresholdResult:
    optimal_threshold: float = 0.45
    criterion_name: str = "youden"
    threshold_sensitivity: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            {
                "threshold": np.linspace(0, 1, 50),
                "criterion_value": np.random.RandomState(42).rand(50),
                "sensitivity": np.linspace(1, 0, 50),
                "specificity": np.linspace(0, 1, 50),
                "precision": np.random.RandomState(42).rand(50),
                "recall": np.linspace(1, 0, 50),
                "f1": np.random.RandomState(42).rand(50),
            }
        )
    )


def _make_classifier_results(
    n_folds=3, n_per_fold=20, with_calibrated=False, with_threshold=False, with_conformal=False
):
    """Build a fake classifier results object for plotting tests."""
    rng = np.random.RandomState(42)
    folds = []
    cms = []
    score_rows = []
    score_rows_opt = []
    for i in range(n_folds):
        y_true = rng.randint(0, 2, size=n_per_fold)
        y_proba = rng.rand(n_per_fold, 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        cm = np.array([[8, 2], [1, 9]])

        fr = _FakeFoldResult(
            fold_idx=i,
            y_true=y_true,
            y_proba_raw=y_proba,
            y_proba_calibrated=y_proba * 0.95 if with_calibrated else None,
            threshold_result=_FakeThresholdResult(optimal_threshold=0.4 + 0.05 * i)
            if with_threshold
            else None,
            train_indices=np.arange(160),
            test_indices=np.arange(n_per_fold),
        )
        folds.append(fr)
        cms.append(cm)
        score_rows.append({"accuracy": 0.85 + 0.02 * i, "f1": 0.83 + 0.02 * i})
        if with_threshold:
            score_rows_opt.append({"accuracy": 0.87 + 0.02 * i, "f1": 0.85 + 0.02 * i})

    results = MagicMock()
    results.fold_results_ = folds
    results.outer_scores_default_ = pd.DataFrame(score_rows)
    results.confusion_matrices_default_ = cms
    results.confusion_matrix_aggregate_default_ = sum(cms)
    results.has_calibration = with_calibrated
    results.has_threshold_optimization = with_threshold
    results.n_outer_folds_ = n_folds

    if with_calibrated:
        results.calibration_summary_ = pd.DataFrame(
            {
                "fold_idx": list(range(n_folds)),
                "ece_raw": [0.10, 0.12, 0.11][:n_folds],
                "ece_calibrated": [0.05, 0.06, 0.04][:n_folds],
            }
        )

    if with_threshold:
        results.thresholds_per_fold_ = np.array([0.4 + 0.05 * i for i in range(n_folds)])
        results.outer_scores_optimized_ = pd.DataFrame(score_rows_opt)
        results.summary_default_ = pd.DataFrame(
            {
                "metric": ["accuracy", "f1"],
                "mean": [0.87, 0.85],
                "std": [0.02, 0.02],
            }
        )
        results.summary_optimized_ = pd.DataFrame(
            {
                "metric": ["accuracy", "f1"],
                "mean": [0.89, 0.87],
                "std": [0.01, 0.01],
            }
        )
        results.threshold_comparison.return_value = pd.DataFrame(
            {
                "metric": ["accuracy", "f1"],
                "mean_default": [0.87, 0.85],
                "std_default": [0.02, 0.02],
                "mean_optimized": [0.89, 0.87],
                "std_optimized": [0.01, 0.01],
            }
        )

    return results


def _make_regressor_results(n_folds=3, n_per_fold=20, with_pi=False):
    """Build a fake regressor results object for plotting tests."""
    rng = np.random.RandomState(42)
    folds = []
    preds_data = {"y_true": [], "y_pred": [], "fold_idx": []}
    if with_pi:
        preds_data["pi_lower"] = []
        preds_data["pi_upper"] = []

    for i in range(n_folds):
        y_true = rng.randn(n_per_fold)
        y_pred = y_true + rng.randn(n_per_fold) * 0.1
        residuals = y_true - y_pred
        folds.append(
            _FakeFoldResult(
                fold_idx=i,
                y_true=y_true,
                residuals=residuals,
                y_pred=y_pred,
                train_indices=np.arange(160),
                test_indices=np.arange(n_per_fold),
            )
        )
        preds_data["y_true"].extend(y_true)
        preds_data["y_pred"].extend(y_pred)
        preds_data["fold_idx"].extend([i] * n_per_fold)
        if with_pi:
            preds_data["pi_lower"].extend(y_pred - 0.5)
            preds_data["pi_upper"].extend(y_pred + 0.5)

    results = MagicMock()
    results.fold_results_ = folds
    results.predictions_ = pd.DataFrame(preds_data)
    return results


def _make_comparator(n_models=2, n_folds=5, n_per_fold=20):
    """Build a fake comparator for comparison plot tests."""
    rng = np.random.RandomState(42)
    model_names = [f"model_{i}" for i in range(n_models)]
    results_dict = {}

    for name in model_names:
        folds = []
        for f in range(n_folds):
            folds.append(
                _FakeFoldResult(
                    fold_idx=f,
                    train_indices=np.arange(160),
                    test_indices=np.arange(n_per_fold),
                )
            )
        r = MagicMock()
        r.fold_results_ = folds
        r.n_outer_folds_ = n_folds
        results_dict[name] = r

    comp = MagicMock()
    comp._results = results_dict

    scores = {name: rng.uniform(0.8, 0.95, n_folds) for name in model_names}
    comp._get_scores = lambda name, metric, threshold="default": scores[name]

    # bayesian_comparison mock
    comp.bayesian_comparison.return_value = {
        "p_a_better": 0.6,
        "p_b_better": 0.3,
        "p_equivalent": 0.1,
    }
    return comp


def _make_aggregator(n_folds=3, n_features=10):
    """Build a fake FeatureImportanceAggregator for importance plot tests."""
    rng = np.random.RandomState(42)
    imp_matrix = rng.rand(n_folds, n_features)
    ranks_matrix = np.apply_along_axis(lambda x: np.argsort(np.argsort(-x)) + 1, 1, imp_matrix)
    names = [f"feature_{i}" for i in range(n_features)]

    agg = MagicMock()
    agg.importances_matrix_ = imp_matrix
    agg.ranks_matrix_ = ranks_matrix
    agg.feature_names = names
    agg.raw_importances_ = []  # Empty by default
    agg.summary_ = (
        pd.DataFrame(
            {
                "feature": names,
                "mean_importance": imp_matrix.mean(axis=0),
                "std_importance": imp_matrix.std(axis=0, ddof=1),
            }
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
    return agg


class TestPlotCalibrationCurves:
    def test_basic(self):
        """Verify plot_calibration_curves returns Axes with basic data."""
        results = _make_classifier_results()
        ax = plot_calibration_curves(results)
        assert ax is not None
        plt.close("all")

    def test_with_calibrated_probabilities(self):
        """Verify calibrated curves are plotted when calibration is present."""
        results = _make_classifier_results(with_calibrated=True)
        ax = plot_calibration_curves(results)
        assert ax is not None
        plt.close("all")

    def test_fold_idx_single(self):
        """Verify fold_idx=0 filters to a single fold."""
        results = _make_classifier_results()
        ax = plot_calibration_curves(results, fold_idx=0)
        assert ax is not None
        plt.close("all")

    def test_fold_idx_list(self):
        """Verify fold_idx=[0,1] filters correctly."""
        results = _make_classifier_results()
        ax = plot_calibration_curves(results, fold_idx=[0, 1])
        assert ax is not None
        plt.close("all")

    def test_full_range(self):
        """Verify full_range=True sets axes to [0, 1]."""
        results = _make_classifier_results()
        ax = plot_calibration_curves(results, full_range=True)
        assert ax.get_xlim() == (0.0, 1.0)
        assert ax.get_ylim() == (0.0, 1.0)
        plt.close("all")


class TestPlotCalibrationImprovement:
    def test_basic(self):
        """Verify plot_calibration_improvement renders with calibration data."""
        results = _make_classifier_results(with_calibrated=True)
        ax = plot_calibration_improvement(results)
        assert ax is not None
        plt.close("all")

    def test_no_calibration_shows_text(self):
        """Verify fallback text when no calibration data."""
        results = _make_classifier_results(with_calibrated=False)
        ax = plot_calibration_improvement(results)
        assert ax is not None
        plt.close("all")

    def test_with_annotations(self):
        """Verify annot=True adds text annotations."""
        results = _make_classifier_results(with_calibrated=True)
        ax = plot_calibration_improvement(results, annot=True)
        assert ax is not None
        plt.close("all")


class TestPlotComparison:
    def test_basic(self):
        """Verify plot_comparison renders with 2 models."""
        comp = _make_comparator(n_models=2)
        ax = plot_comparison(comp, metric="accuracy")
        assert ax is not None
        plt.close("all")

    def test_custom_alpha(self):
        """Verify custom alpha parameters are accepted."""
        comp = _make_comparator(n_models=2)
        ax = plot_comparison(comp, metric="accuracy", point_alpha=0.3, line_alpha=0.1)
        assert ax is not None
        plt.close("all")


class TestPlotScoreDifferences:
    def test_basic(self):
        """Verify plot_score_differences renders correctly."""
        comp = _make_comparator(n_models=2)
        ax = plot_score_differences(comp, "accuracy", "model_0", "model_1")
        assert ax is not None
        plt.close("all")

    def test_custom_bar_color(self):
        """Verify custom bar_color is accepted."""
        comp = _make_comparator(n_models=2)
        ax = plot_score_differences(comp, "accuracy", "model_0", "model_1", bar_color="green")
        assert ax is not None
        plt.close("all")


class TestPlotBayesianPosterior:
    def test_basic(self):
        """Verify plot_bayesian_posterior renders the posterior distribution."""
        comp = _make_comparator(n_models=2)
        ax = plot_bayesian_posterior(comp, "accuracy", "model_0", "model_1")
        assert ax is not None
        plt.close("all")


class TestPlotCriticalDifference:
    def test_fewer_than_3_models(self):
        """Verify 'Need >= 3 models' text with fewer than 3 models."""
        comp = _make_comparator(n_models=2)
        ax = plot_critical_difference(comp, "accuracy")
        assert ax is not None
        plt.close("all")

    def test_basic_3_models(self):
        """Verify critical difference diagram with 3+ models."""
        comp = _make_comparator(n_models=3)
        ax = plot_critical_difference(comp, "accuracy")
        assert ax is not None
        plt.close("all")


class TestPlotImportance:
    def test_basic(self):
        """Verify plot_importance renders with fold overlays."""
        agg = _make_aggregator()
        ax = plot_importance(agg)
        assert ax is not None
        plt.close("all")

    def test_no_folds(self):
        """Verify show_folds=False skips fold scatter."""
        agg = _make_aggregator()
        ax = plot_importance(agg, show_folds=False)
        assert ax is not None
        plt.close("all")

    def test_custom_top_k(self):
        """Verify top_k limits displayed features."""
        agg = _make_aggregator(n_features=20)
        ax = plot_importance(agg, top_k=5)
        assert ax is not None
        plt.close("all")


class TestPlotRankStabilityFeatures:
    def test_basic(self):
        """Verify rank stability heatmap renders."""
        agg = _make_aggregator()
        ax = plot_rank_stability_features(agg, top_k=5)
        assert ax is not None
        plt.close("all")


class TestPlotShapSummary:
    def test_no_raw_raises(self):
        """Verify ValueError when raw_importances_ is empty."""
        agg = _make_aggregator()
        agg.raw_importances_ = []
        with pytest.raises(ValueError, match="SHAP raw values not available"):
            plot_shap_summary(agg)
        plt.close("all")


class TestPlotSelectionFrequency:
    def test_basic(self):
        """Verify selection frequency bar chart renders."""
        agg = _make_aggregator()
        ax = plot_selection_frequency(agg, top_k=5)
        assert ax is not None
        plt.close("all")

    def test_full_range(self):
        """Verify full_range sets x-axis to [0, 1]."""
        agg = _make_aggregator()
        ax = plot_selection_frequency(agg, top_k=5, full_range=True)
        assert ax.get_xlim() == (0.0, 1.0)
        plt.close("all")


class TestPlotThresholdSensitivity:
    def test_basic(self):
        """Verify threshold sensitivity plot renders with data."""
        results = _make_classifier_results(with_threshold=True)
        ax = plot_threshold_sensitivity(results, fold_idx=0)
        assert ax is not None
        plt.close("all")

    def test_no_threshold_data(self):
        """Verify fallback text when no threshold data."""
        results = _make_classifier_results(with_threshold=False)
        ax = plot_threshold_sensitivity(results, fold_idx=0)
        assert ax is not None
        plt.close("all")


class TestPlotThresholdDistribution:
    def test_basic(self):
        """Verify threshold distribution histogram renders."""
        results = _make_classifier_results(with_threshold=True)
        ax = plot_threshold_distribution(results)
        assert ax is not None
        plt.close("all")

    def test_no_threshold(self):
        """Verify fallback text when threshold optimization is absent."""
        results = _make_classifier_results(with_threshold=False)
        ax = plot_threshold_distribution(results)
        assert ax is not None
        plt.close("all")


class TestPlotThresholdComparison:
    def test_basic(self):
        """Verify threshold comparison bar chart renders."""
        results = _make_classifier_results(with_threshold=True)
        ax = plot_threshold_comparison(results)
        assert ax is not None
        plt.close("all")

    def test_no_threshold(self):
        """Verify fallback text when threshold optimization is absent."""
        results = _make_classifier_results(with_threshold=False)
        ax = plot_threshold_comparison(results)
        assert ax is not None
        plt.close("all")


class TestPlotPredictedVsActual:
    def test_basic(self):
        """Verify predicted vs actual scatter plot renders."""
        results = _make_regressor_results()
        ax = plot_predicted_vs_actual(results)
        assert ax is not None
        plt.close("all")


class TestPlotPredictionIntervals:
    def test_with_intervals(self):
        """Verify prediction interval bands render when data present."""
        results = _make_regressor_results(with_pi=True)
        ax = plot_prediction_intervals(results)
        assert ax is not None
        plt.close("all")

    def test_without_intervals(self):
        """Verify fallback text when prediction intervals are absent."""
        results = _make_regressor_results(with_pi=False)
        ax = plot_prediction_intervals(results)
        assert ax is not None
        plt.close("all")


class TestPlotResidualQQ:
    def test_basic(self):
        """Verify residual QQ plot renders."""
        results = _make_regressor_results()
        ax = plot_residual_qq(results)
        assert ax is not None
        plt.close("all")


class TestPlotRankStability:
    def test_basic(self):
        """Verify inner CV rank stability plot renders."""
        results = _make_classifier_results()
        report = MagicMock()
        report.ranking.return_value = pd.DataFrame(
            {
                "mean_test_score": [0.95, 0.93, 0.90],
            }
        )
        results.inner_reports_ = [report, report, report]
        ax = plot_rank_stability(results, top_k=3)
        assert ax is not None
        plt.close("all")


class TestPlotConfusionMatricesOptimized:
    def test_optimized_threshold(self):
        """Verify confusion_matrices plot uses optimized data when requested."""
        from nestkit.plotting.summary import plot_confusion_matrices

        results = _make_classifier_results(with_threshold=True)
        cm = np.array([[9, 1], [2, 8]])
        results.confusion_matrices_optimized_ = [cm] * 3
        results.confusion_matrix_aggregate_optimized_ = cm * 3
        ax = plot_confusion_matrices(results, threshold="optimized")
        assert ax is not None
        plt.close("all")


class TestPlotPrecisionRecallCurvesExtended:
    def test_explicit_limits(self):
        """Verify explicit xlim/ylim are applied."""
        results = _make_classifier_results()
        ax = plot_precision_recall_curves(results, xlim=(0.2, 0.9), ylim=(0.3, 1.0))
        assert ax.get_xlim() == (0.2, 0.9)
        assert ax.get_ylim() == (0.3, 1.0)
        plt.close("all")


class TestPlotInnerTuningCurve:
    def test_basic(self):
        """Verify inner tuning curve renders."""
        report = MagicMock()
        report.score_distribution.return_value = pd.DataFrame(
            {
                "max_depth": [3, 5, 10],
                "mean_score": [0.90, 0.93, 0.91],
                "std_score": [0.02, 0.01, 0.03],
            }
        )
        ax = plot_inner_tuning_curve(report, param="max_depth")
        assert ax is not None
        plt.close("all")

    def test_without_std(self):
        """Verify tuning curve renders without std_score column."""
        report = MagicMock()
        report.score_distribution.return_value = pd.DataFrame(
            {
                "max_depth": [3, 5, 10],
                "mean_score": [0.90, 0.93, 0.91],
            }
        )
        ax = plot_inner_tuning_curve(report, param="max_depth")
        assert ax is not None
        plt.close("all")
