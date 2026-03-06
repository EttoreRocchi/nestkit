"""Classifier-specific results containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report

from nestkit.results._base import _BaseNestedCVResults
from nestkit.thresholding.results import ThresholdResult


@dataclass
class ClassifierOuterFoldResult:
    """Result of a single outer fold evaluation (classification)."""

    # Core (always)
    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    best_params: dict
    best_inner_score: float
    inner_cv_results: dict
    fit_time: float
    score_time: float
    fitted_estimator: BaseEstimator | None

    # Predictions (always)
    y_true: np.ndarray
    y_proba_raw: np.ndarray
    y_pred_default: np.ndarray
    outer_scores_default: dict = field(default_factory=dict)
    confusion_matrix_default: np.ndarray = field(default_factory=lambda: np.array([]))

    # [OPT-CAL]
    y_proba_calibrated: np.ndarray | None = None
    calibration_method: str | None = None
    calibrator: Any | None = None
    oof_calibration_diagnostics: dict | None = None

    # [OPT-THR]
    y_pred_optimized: np.ndarray | None = None
    outer_scores_optimized: dict | None = None
    confusion_matrix_optimized: np.ndarray | None = None
    threshold_result: ThresholdResult | None = None


class ClassifierResults(_BaseNestedCVResults):
    """Aggregated nested CV results for classification."""

    def __init__(
        self,
        n_outer_folds: int,
        feature_names: list[str] | None = None,
        original_index: Any | None = None,
    ):
        super().__init__(n_outer_folds, feature_names, original_index)

    @property
    def has_calibration(self) -> bool:
        if not self.fold_results_:
            return False
        return self.fold_results_[0].calibration_method is not None

    @property
    def has_threshold_optimization(self) -> bool:
        if not self.fold_results_:
            return False
        return self.fold_results_[0].threshold_result is not None

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        self.best_params_per_fold_ = [fr.best_params for fr in self.fold_results_]

        # Param stability
        self._compute_param_stability()

        # Inner reports
        from nestkit.inner.tuning_report import InnerCVReport

        self.inner_reports_ = [
            InnerCVReport(fr.inner_cv_results, fr.fold_idx) for fr in self.fold_results_
        ]

        # Default scores
        self.outer_scores_default_ = pd.DataFrame(
            [fr.outer_scores_default for fr in self.fold_results_]
        )
        self.summary_default_ = self._compute_summary(self.outer_scores_default_)

        # Confusion matrices
        self.confusion_matrices_default_ = [
            fr.confusion_matrix_default for fr in self.fold_results_
        ]
        self.confusion_matrix_aggregate_default_ = sum(self.confusion_matrices_default_)

        # Predictions DataFrame
        self._build_predictions_df()

        # Generalization gap
        self._compute_generalization_gap()

        # Calibration attributes
        if self.has_calibration:
            self._compute_calibration_attributes()

        # Threshold attributes
        if self.has_threshold_optimization:
            self._compute_threshold_attributes()

    def _compute_param_stability(self) -> None:
        if not self.best_params_per_fold_:
            self.param_stability_ = pd.DataFrame()
            return
        all_params = set()
        for p in self.best_params_per_fold_:
            all_params.update(p.keys())

        rows = []
        for param in sorted(all_params):
            values = [p.get(param) for p in self.best_params_per_fold_]
            from collections import Counter

            counts = Counter(values)
            mode_val, mode_count = counts.most_common(1)[0]
            rows.append(
                {
                    "parameter": param,
                    "mode": mode_val,
                    "nunique": len(counts),
                    "agreement_rate": mode_count / len(values),
                }
            )
        self.param_stability_ = pd.DataFrame(rows)

    def _build_predictions_df(self) -> None:
        dfs = []
        for fr in self.fold_results_:
            fold_df = pd.DataFrame(
                {
                    "y_true": fr.y_true,
                    "y_pred_default": fr.y_pred_default,
                    "fold_idx": fr.fold_idx,
                }
            )
            # Add raw probabilities
            if fr.y_proba_raw.ndim == 2:
                for c in range(fr.y_proba_raw.shape[1]):
                    fold_df[f"y_proba_raw_{c}"] = fr.y_proba_raw[:, c]
            else:
                fold_df["y_proba_raw"] = fr.y_proba_raw

            # Calibrated probabilities
            if fr.y_proba_calibrated is not None:
                if fr.y_proba_calibrated.ndim == 2:
                    for c in range(fr.y_proba_calibrated.shape[1]):
                        fold_df[f"y_proba_cal_{c}"] = fr.y_proba_calibrated[:, c]
                else:
                    fold_df["y_proba_cal"] = fr.y_proba_calibrated

            # Optimized predictions
            if fr.y_pred_optimized is not None:
                fold_df["y_pred_optimized"] = fr.y_pred_optimized

            # Set original index if available
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
            for metric, val in fr.outer_scores_default.items():
                row[f"outer_{metric}"] = val
                row[f"gap_{metric}"] = fr.best_inner_score - val
            rows.append(row)
        self.generalization_gap_ = pd.DataFrame(rows)

    def _compute_calibration_attributes(self) -> None:
        rows = []
        for fr in self.fold_results_:
            if fr.oof_calibration_diagnostics:
                rows.append({"fold_idx": fr.fold_idx, **fr.oof_calibration_diagnostics})
        self.calibration_summary_ = pd.DataFrame(rows) if rows else pd.DataFrame()

        if not self.calibration_summary_.empty:
            improvement_rows = []
            for _, row in self.calibration_summary_.iterrows():
                imp = {"fold_idx": row["fold_idx"]}
                if "ece_raw" in row and "ece_calibrated" in row:
                    imp["delta_ece"] = row["ece_raw"] - row["ece_calibrated"]
                if "brier_raw" in row and "brier_calibrated" in row:
                    imp["delta_brier"] = row["brier_raw"] - row["brier_calibrated"]
                improvement_rows.append(imp)
            self.calibration_improvement_ = pd.DataFrame(improvement_rows)
        else:
            self.calibration_improvement_ = pd.DataFrame()

    def _compute_threshold_attributes(self) -> None:
        self.outer_scores_optimized_ = pd.DataFrame(
            [fr.outer_scores_optimized for fr in self.fold_results_ if fr.outer_scores_optimized]
        )
        self.summary_optimized_ = self._compute_summary(self.outer_scores_optimized_)

        self.thresholds_per_fold_ = np.array(
            [
                fr.threshold_result.optimal_threshold
                for fr in self.fold_results_
                if fr.threshold_result
            ]
        )
        if len(self.thresholds_per_fold_) > 0:
            self.threshold_stability_ = {
                "mean": float(np.mean(self.thresholds_per_fold_)),
                "std": float(np.std(self.thresholds_per_fold_, ddof=1)),
                "cv": float(
                    np.std(self.thresholds_per_fold_, ddof=1)
                    / (np.mean(self.thresholds_per_fold_) + 1e-12)
                ),
                "range": float(np.ptp(self.thresholds_per_fold_)),
            }
        else:
            self.threshold_stability_ = {}

        self.confusion_matrices_optimized_ = [
            fr.confusion_matrix_optimized
            for fr in self.fold_results_
            if fr.confusion_matrix_optimized is not None
        ]
        if self.confusion_matrices_optimized_:
            self.confusion_matrix_aggregate_optimized_ = sum(self.confusion_matrices_optimized_)
        else:
            self.confusion_matrix_aggregate_optimized_ = np.array([])

    def threshold_comparison(self) -> pd.DataFrame:
        """Side-by-side comparison of default vs optimized threshold performance."""
        if not self.has_threshold_optimization:
            raise ValueError("Threshold optimization was not enabled for this run.")
        default = self.summary_default_.copy().rename(
            columns={"mean": "mean_default", "std": "std_default"}
        )
        optimized = self.summary_optimized_.copy().rename(
            columns={"mean": "mean_optimized", "std": "std_optimized"}
        )
        return pd.merge(
            default[["metric", "mean_default", "std_default"]],
            optimized[["metric", "mean_optimized", "std_optimized"]],
            on="metric",
        )

    def calibration_report(self) -> pd.DataFrame:
        """Detailed calibration diagnostics per fold."""
        if not self.has_calibration:
            raise ValueError("Calibration was not enabled for this run.")
        return self.calibration_summary_.copy()

    def classification_report_pooled(self, threshold: str = "default") -> str:
        """sklearn-style classification report on pooled OOF predictions."""
        y_true = self.predictions_["y_true"].values
        if threshold == "optimized":
            if not self.has_threshold_optimization:
                raise ValueError("Threshold optimization was not enabled.")
            y_pred = self.predictions_["y_pred_optimized"].values
        else:
            y_pred = self.predictions_["y_pred_default"].values
        return classification_report(y_true, y_pred)
