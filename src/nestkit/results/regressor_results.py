"""Regressor-specific results containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from nestkit.results._base import _BaseNestedCVResults


@dataclass
class RegressorOuterFoldResult:
    """Result of a single outer fold evaluation (regression)."""

    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    best_params: dict
    best_inner_score: float
    inner_cv_results: dict
    fit_time: float
    score_time: float
    fitted_estimator: BaseEstimator | None

    y_true: np.ndarray
    y_pred: np.ndarray
    outer_scores: dict = field(default_factory=dict)
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))

    # Optional prediction intervals
    prediction_interval_lower: np.ndarray | None = None
    prediction_interval_upper: np.ndarray | None = None
    coverage: float | None = None


class RegressorResults(_BaseNestedCVResults):
    """Aggregated nested CV results for regression."""

    def __init__(
        self,
        n_outer_folds: int,
        feature_names: list[str] | None = None,
        original_index: Any | None = None,
    ):
        super().__init__(n_outer_folds, feature_names, original_index)

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        self.best_params_per_fold_ = [fr.best_params for fr in self.fold_results_]

        from nestkit.inner.tuning_report import InnerCVReport

        self.inner_reports_ = [
            InnerCVReport(fr.inner_cv_results, fr.fold_idx) for fr in self.fold_results_
        ]

        self.outer_scores_default_ = pd.DataFrame([fr.outer_scores for fr in self.fold_results_])
        self.summary_default_ = self._compute_summary(self.outer_scores_default_)

        self._build_predictions_df()
        self._compute_generalization_gap()
        self._compute_residual_stats()

    def _build_predictions_df(self) -> None:
        dfs = []
        for fr in self.fold_results_:
            fold_df = pd.DataFrame(
                {
                    "y_true": fr.y_true,
                    "y_pred": fr.y_pred,
                    "residual": fr.residuals,
                    "fold_idx": fr.fold_idx,
                }
            )
            if fr.prediction_interval_lower is not None:
                fold_df["pi_lower"] = fr.prediction_interval_lower
                fold_df["pi_upper"] = fr.prediction_interval_upper

            if self._original_index is not None:
                fold_df.index = self._original_index[fr.test_indices]
            else:
                fold_df.index = fr.test_indices

            dfs.append(fold_df)
        self.predictions_ = pd.concat(dfs).sort_index()

    def _compute_generalization_gap(self) -> None:
        rows = []
        for fr in self.fold_results_:
            row = {"fold_idx": fr.fold_idx, "best_inner_score": fr.best_inner_score}
            for metric, val in fr.outer_scores.items():
                row[f"outer_{metric}"] = val
                row[f"gap_{metric}"] = fr.best_inner_score - val
            rows.append(row)
        self.generalization_gap_ = pd.DataFrame(rows)

    def _compute_residual_stats(self) -> None:
        all_residuals = np.concatenate([fr.residuals for fr in self.fold_results_])
        self.residual_stats_ = {
            "mean": float(np.mean(all_residuals)),
            "std": float(np.std(all_residuals, ddof=1)),
            "median": float(np.median(all_residuals)),
            "skewness": float(_skewness(all_residuals)),
            "kurtosis": float(_kurtosis(all_residuals)),
        }

        # Prediction interval coverage
        coverages = [fr.coverage for fr in self.fold_results_ if fr.coverage is not None]
        if coverages:
            self.prediction_interval_coverage_ = {
                "mean": float(np.mean(coverages)),
                "per_fold": coverages,
            }
        else:
            self.prediction_interval_coverage_ = None


def _skewness(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return (n / ((n - 1) * (n - 2))) * np.sum(((x - m) / s) ** 3)


def _kurtosis(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((x - m) / s) ** 4) - 3 * (
        n - 1
    ) ** 2 / ((n - 2) * (n - 3))
