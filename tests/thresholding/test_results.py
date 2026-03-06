"""Tests for ThresholdResult dataclass."""

import numpy as np
import pandas as pd

from nestkit.thresholding.results import ThresholdResult


class TestThresholdResult:
    def test_construction_fold_specific(self):
        result = ThresholdResult(
            strategy="fold_specific",
            optimal_threshold=0.42,
            criterion_name="youden_j",
            criterion_value_at_optimum=0.65,
            fold_thresholds=np.array([0.40, 0.43, 0.44]),
            fold_threshold_std=0.02,
        )
        assert result.strategy == "fold_specific"
        assert result.optimal_threshold == 0.42
        assert result.fold_thresholds is not None
        assert len(result.fold_thresholds) == 3

    def test_construction_pooled(self):
        result = ThresholdResult(
            strategy="pooled",
            optimal_threshold=0.5,
            criterion_name="f_1.0",
            criterion_value_at_optimum=0.8,
        )
        assert result.strategy == "pooled"
        assert result.fold_thresholds is None
        assert result.fold_threshold_std is None

    def test_default_threshold_sensitivity_empty_dataframe(self):
        result = ThresholdResult(
            strategy="pooled",
            optimal_threshold=0.5,
            criterion_name="youden_j",
            criterion_value_at_optimum=0.7,
        )
        assert isinstance(result.threshold_sensitivity, pd.DataFrame)
        assert len(result.threshold_sensitivity) == 0

    def test_extreme_threshold_zero(self):
        result = ThresholdResult(
            strategy="pooled",
            optimal_threshold=0.0,
            criterion_name="youden_j",
            criterion_value_at_optimum=0.0,
        )
        assert result.optimal_threshold == 0.0

    def test_extreme_threshold_one(self):
        result = ThresholdResult(
            strategy="pooled",
            optimal_threshold=1.0,
            criterion_name="youden_j",
            criterion_value_at_optimum=0.0,
        )
        assert result.optimal_threshold == 1.0

    def test_threshold_sensitivity_with_data(self):
        sensitivity = pd.DataFrame(
            {
                "threshold": [0.3, 0.5, 0.7],
                "criterion_value": [0.6, 0.8, 0.7],
                "sensitivity": [0.9, 0.8, 0.6],
                "specificity": [0.5, 0.7, 0.9],
            }
        )
        result = ThresholdResult(
            strategy="pooled",
            optimal_threshold=0.5,
            criterion_name="f_1.0",
            criterion_value_at_optimum=0.8,
            threshold_sensitivity=sensitivity,
        )
        assert len(result.threshold_sensitivity) == 3
        assert "threshold" in result.threshold_sensitivity.columns
