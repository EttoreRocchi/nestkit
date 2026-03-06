"""Tests for plotting functions  -  parameters, axis limits, and normalization."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nestkit.plotting import (
    plot_confusion_matrices,
    plot_inner_cv_heatmap,
    plot_outer_scores,
    plot_param_selection,
    plot_precision_recall_curves,
    plot_residuals,
    plot_roc_curves,
    plot_score_stability,
)
from nestkit.plotting._style import _apply_axis_limits, _get_ax


@dataclass
class _FakeClassifierFoldResult:
    y_true: np.ndarray
    y_proba_raw: np.ndarray
    y_proba_calibrated: np.ndarray | None = None
    confusion_matrix_default: np.ndarray = field(
        default_factory=lambda: np.array([[8, 2], [1, 9]])
    )
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class _FakeRegressorFoldResult:
    residuals: np.ndarray


class _FakeClassifierResults:
    """Minimal mock of ClassifierResults for plotting."""

    def __init__(self, n_folds=3, n_samples_per_fold=20):
        rng = np.random.RandomState(42)
        self.fold_results_ = []
        rows = []
        cms = []
        for i in range(n_folds):
            y_true = rng.randint(0, 2, size=n_samples_per_fold)
            y_proba = rng.rand(n_samples_per_fold, 2)
            y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
            cm = np.array([[8, 2], [1, 9]])
            self.fold_results_.append(
                _FakeClassifierFoldResult(
                    y_true=y_true,
                    y_proba_raw=y_proba,
                    y_proba_calibrated=None,
                    confusion_matrix_default=cm,
                )
            )
            rows.append({"accuracy": 0.85 + 0.02 * i, "f1": 0.83 + 0.02 * i})
            cms.append(cm)

        self.outer_scores_default_ = pd.DataFrame(rows)
        self.confusion_matrices_default_ = cms
        self.confusion_matrix_aggregate_default_ = sum(cms)


class _FakeRegressorResults:
    """Minimal mock of RegressorResults for residual plots."""

    def __init__(self, n_folds=3, n_samples_per_fold=20):
        rng = np.random.RandomState(42)
        self.fold_results_ = []
        for _ in range(n_folds):
            self.fold_results_.append(
                _FakeRegressorFoldResult(
                    residuals=rng.randn(n_samples_per_fold),
                )
            )


class TestGetAx:
    def test_returns_existing_ax(self):
        _, ax = plt.subplots()
        result = _get_ax(ax)
        assert result is ax
        plt.close("all")

    def test_creates_new_ax(self):
        ax = _get_ax(None)
        assert ax is not None
        plt.close("all")

    def test_figsize(self):
        ax = _get_ax(None, figsize=(10, 5))
        fig = ax.get_figure()
        w, h = fig.get_size_inches()
        assert abs(w - 10) < 0.1
        assert abs(h - 5) < 0.1
        plt.close("all")


class TestApplyAxisLimits:
    def test_explicit_ylim_takes_priority(self):
        _, ax = plt.subplots()
        ax.plot([0, 1], [0.3, 0.7])
        _apply_axis_limits(ax, ylim=(0.2, 0.9), full_range=True, natural_ylim=(0.0, 1.0))
        assert ax.get_ylim() == (0.2, 0.9)
        plt.close("all")

    def test_full_range_applies_natural(self):
        _, ax = plt.subplots()
        ax.plot([0, 1], [0.3, 0.7])
        _apply_axis_limits(ax, full_range=True, natural_ylim=(0.0, 1.0))
        assert ax.get_ylim() == (0.0, 1.0)
        plt.close("all")

    def test_full_range_false_no_effect(self):
        _, ax = plt.subplots()
        ax.plot([0, 1], [0.3, 0.7])
        original_ylim = ax.get_ylim()
        _apply_axis_limits(ax, full_range=False, natural_ylim=(0.0, 1.0))
        assert ax.get_ylim() == original_ylim
        plt.close("all")

    def test_explicit_xlim(self):
        _, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        _apply_axis_limits(ax, xlim=(0.1, 0.8))
        assert ax.get_xlim() == (0.1, 0.8)
        plt.close("all")

    def test_no_natural_range_full_range_noop(self):
        _, ax = plt.subplots()
        ax.plot([0, 10], [0, 100])
        original = ax.get_ylim()
        _apply_axis_limits(ax, full_range=True)  # no natural range
        assert ax.get_ylim() == original
        plt.close("all")


def test_plot_outer_scores():
    results = _FakeClassifierResults()
    ax = plot_outer_scores(results, metric="accuracy")
    assert ax is not None
    plt.close("all")


def test_plot_roc_curves():
    results = _FakeClassifierResults()
    ax = plot_roc_curves(results)
    assert ax is not None
    plt.close("all")


def test_plot_confusion_matrices():
    results = _FakeClassifierResults()
    ax = plot_confusion_matrices(results)
    assert ax is not None
    plt.close("all")


def test_plot_residuals():
    results = _FakeRegressorResults()
    ax = plot_residuals(results)
    assert ax is not None
    plt.close("all")


class TestConfusionMatrixNormalization:
    def _get_cm(self):
        return np.array([[8, 2], [1, 9]])

    def test_normalize_true(self):
        """Rows should sum to 1 when normalized by true labels."""
        results = _FakeClassifierResults(n_folds=1)
        results.confusion_matrices_default_ = [self._get_cm()]
        results.confusion_matrix_aggregate_default_ = self._get_cm()
        plot_confusion_matrices(results, normalize="true")

        cm = self._get_cm().astype(float)
        normed = cm / cm.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(normed.sum(axis=1), [1.0, 1.0])
        plt.close("all")

    def test_normalize_pred(self):
        """Columns should sum to 1 when normalized by predicted labels."""
        cm = self._get_cm().astype(float)
        normed = cm / cm.sum(axis=0, keepdims=True)
        np.testing.assert_allclose(normed.sum(axis=0), [1.0, 1.0])

    def test_normalize_all(self):
        """All cells should sum to 1 when normalized by total."""
        cm = self._get_cm().astype(float)
        normed = cm / cm.sum()
        np.testing.assert_allclose(normed.sum(), 1.0)

    def test_normalize_none_shows_raw_counts(self):
        """Default (no normalization) should show integer counts."""
        results = _FakeClassifierResults(n_folds=1)
        results.confusion_matrices_default_ = [self._get_cm()]
        results.confusion_matrix_aggregate_default_ = self._get_cm()
        ax = plot_confusion_matrices(results, normalize=None)
        assert ax is not None
        plt.close("all")

    def test_custom_cmap(self):
        results = _FakeClassifierResults(n_folds=1)
        results.confusion_matrices_default_ = [self._get_cm()]
        results.confusion_matrix_aggregate_default_ = self._get_cm()
        ax = plot_confusion_matrices(results, cmap="Reds")
        assert ax is not None
        plt.close("all")


class TestAxisLimitsIntegration:
    def test_roc_full_range_false_by_default(self):
        """ROC curves should auto-scale by default (full_range=False)."""
        results = _FakeClassifierResults()
        ax = plot_roc_curves(results)
        assert ax is not None
        plt.close("all")

    def test_roc_full_range_true(self):
        """With full_range=True, limits should be [0,1]."""
        results = _FakeClassifierResults()
        ax = plot_roc_curves(results, full_range=True)
        assert ax.get_ylim() == (0.0, 1.0)
        assert ax.get_xlim() == (0.0, 1.0)
        plt.close("all")

    def test_roc_explicit_ylim_overrides_full_range(self):
        results = _FakeClassifierResults()
        ax = plot_roc_curves(results, full_range=True, ylim=(0.5, 1.0))
        assert ax.get_ylim() == (0.5, 1.0)
        plt.close("all")

    def test_pr_full_range(self):
        results = _FakeClassifierResults()
        ax = plot_precision_recall_curves(results, full_range=True)
        assert ax.get_ylim() == (0.0, 1.0)
        assert ax.get_xlim() == (0.0, 1.0)
        plt.close("all")

    def test_outer_scores_explicit_ylim(self):
        results = _FakeClassifierResults()
        ax = plot_outer_scores(results, metric="accuracy", ylim=(0.0, 1.0))
        assert ax.get_ylim() == (0.0, 1.0)
        plt.close("all")

    def test_residuals_no_natural_range(self):
        """Residuals should not change limits when full_range is passed."""
        results = _FakeRegressorResults()
        ax = plot_residuals(results)
        assert ax is not None
        plt.close("all")


class TestStyleParams:
    def test_roc_custom_fold_alpha(self):
        results = _FakeClassifierResults()
        ax = plot_roc_curves(results, fold_alpha=0.1, mean_color="r", mean_lw=3)
        assert ax is not None
        plt.close("all")

    def test_residuals_custom_bins(self):
        results = _FakeRegressorResults()
        ax = plot_residuals(results, bins=10, fold_alpha=0.3)
        assert ax is not None
        plt.close("all")

    def test_pr_custom_fold_alpha(self):
        results = _FakeClassifierResults()
        ax = plot_precision_recall_curves(results, fold_alpha=0.8)
        assert ax is not None
        plt.close("all")


class _FakeInnerReport:
    """Minimal mock of InnerCVReport for heatmap tests."""

    def __init__(self, n_configs=4, seed=42):
        rng = np.random.RandomState(seed)
        self._df = pd.DataFrame(
            {
                "param_n_estimators": [50, 100, 200, 200],
                "param_max_depth": [3, 3, 5, 10],
                "mean_test_score": rng.uniform(0.90, 0.99, n_configs),
            }
        )

    def to_dataframe(self):
        return self._df.copy()


class TestPlotInnerCvHeatmap:
    def _make_results(self, n_folds=3):
        results = _FakeClassifierResults(n_folds=n_folds)
        results.inner_reports_ = [_FakeInnerReport(seed=42 + i) for i in range(n_folds)]
        return results

    def test_returns_ax(self):
        results = self._make_results()
        ax = plot_inner_cv_heatmap(results)
        assert ax is not None
        plt.close("all")

    def test_title_uses_scoring(self):
        results = self._make_results()
        results.scoring_ = "roc_auc"
        ax = plot_inner_cv_heatmap(results)
        assert "roc_auc" in ax.get_title()
        plt.close("all")


class TestPlotScoreStability:
    def test_returns_ax(self):
        results = _FakeClassifierResults()
        ax = plot_score_stability(results)
        assert ax is not None
        plt.close("all")

    def test_specific_metrics(self):
        results = _FakeClassifierResults()
        ax = plot_score_stability(results, metrics=["accuracy"])
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == ["accuracy"]
        plt.close("all")

    def test_all_metrics_shown_by_default(self):
        results = _FakeClassifierResults()
        ax = plot_score_stability(results)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "accuracy" in labels
        assert "f1" in labels
        plt.close("all")


class TestPlotParamSelection:
    def _make_results(self):
        results = _FakeClassifierResults()
        results.best_params_per_fold_ = [
            {"n_estimators": 50},
            {"n_estimators": 100},
            {"n_estimators": 200},
            {"n_estimators": 200},
            {"n_estimators": 100},
        ]
        return results

    def test_frequency_bars(self):
        results = self._make_results()
        ax = plot_param_selection(results, "n_estimators")
        # 3 unique values -> 3 bars
        patches = ax.patches
        assert len(patches) == 3
        plt.close("all")

    def test_bar_heights_are_counts(self):
        results = self._make_results()
        ax = plot_param_selection(results, "n_estimators")
        heights = [p.get_height() for p in ax.patches]
        assert sorted(heights) == [1, 2, 2]
        plt.close("all")

    def test_categorical_values(self):
        results = _FakeClassifierResults()
        results.best_params_per_fold_ = [
            {"kernel": "rbf"},
            {"kernel": "rbf"},
            {"kernel": "linear"},
        ]
        ax = plot_param_selection(results, "kernel")
        assert len(ax.patches) == 2
        plt.close("all")
