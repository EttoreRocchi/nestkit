"""Survival-specific results containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from nestkit.results._base import _BaseNestedCVResults


@dataclass
class SurvivalOuterFoldResult:
    """Result of a single outer fold evaluation (survival)."""

    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    best_params: dict
    best_inner_score: float
    inner_cv_results: dict
    fit_time: float
    score_time: float
    fitted_estimator: BaseEstimator | None

    y_event: np.ndarray
    y_duration: np.ndarray
    risk_scores: np.ndarray
    outer_scores: dict = field(default_factory=dict)

    coefficients: dict | None = None

    ibs_eval_times: np.ndarray | None = None


class SurvivalResults(_BaseNestedCVResults):
    """Aggregated nested CV results for survival analysis."""

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
        self._compute_coefficient_stability()

    def _build_predictions_df(self) -> None:
        dfs = []
        for fr in self.fold_results_:
            fold_df = pd.DataFrame(
                {
                    "y_event": fr.y_event,
                    "y_duration": fr.y_duration,
                    "risk_score": fr.risk_scores,
                    "fold_idx": fr.fold_idx,
                }
            )

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
            rows.append(row)
        self.generalization_gap_ = pd.DataFrame(rows)

    def _compute_coefficient_stability(self) -> None:
        """Compute per-feature coefficient and hazard ratio stability."""
        coef_dicts = [fr.coefficients for fr in self.fold_results_ if fr.coefficients]
        if not coef_dicts:
            self.coefficient_stability_ = pd.DataFrame()
            return

        all_features = sorted(set().union(*(d.keys() for d in coef_dicts)))
        rows = []
        for feat in all_features:
            vals = [d.get(feat, np.nan) for d in coef_dicts]
            valid = [v for v in vals if not np.isnan(v)]
            hr_vals = [np.exp(v) for v in valid]
            rows.append(
                {
                    "feature": feat,
                    "coef_mean": float(np.nanmean(vals)),
                    "coef_std": float(np.nanstd(vals, ddof=1)) if len(valid) > 1 else 0.0,
                    "hazard_ratio_mean": float(np.mean(hr_vals)) if hr_vals else np.nan,
                    "hazard_ratio_std": (
                        float(np.std(hr_vals, ddof=1)) if len(hr_vals) > 1 else 0.0
                    ),
                }
            )
        self.coefficient_stability_ = pd.DataFrame(rows)
