"""Tests for thresholding module."""

import numpy as np
import pytest

from nestkit.thresholding.criteria import (
    balanced_accuracy_criterion,
    cost_sensitive,
    f_beta_criterion,
    precision_at_recall,
    youden_j,
)
from nestkit.thresholding.strategies import FoldSpecificThreshold, PooledThreshold


class TestCriteria:
    def test_youden_j(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        j = youden_j(y_true, y_proba, threshold=0.5)
        assert j == pytest.approx(1.0)

    def test_f_beta_criterion(self):
        criterion = f_beta_criterion(beta=1.0)
        assert callable(criterion)
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        score = criterion(y_true, y_proba, 0.5)
        assert isinstance(score, float)

    def test_cost_sensitive(self):
        cost_matrix = [[0, 1], [10, 0]]
        criterion = cost_sensitive(cost_matrix)
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        score = criterion(y_true, y_proba, 0.5)
        assert score == pytest.approx(0.0)  # negative cost = 0

    def test_balanced_accuracy_criterion(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        ba = balanced_accuracy_criterion(y_true, y_proba, 0.5)
        assert ba == pytest.approx(1.0)

    def test_precision_at_recall(self):
        criterion = precision_at_recall(min_recall=1.0)
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.6, 0.4])
        score = criterion(y_true, y_proba, 0.5)
        assert score == pytest.approx(-1.0)


class TestStrategies:
    @pytest.fixture
    def fold_data(self):
        rng = np.random.RandomState(42)
        folds_y = []
        folds_p = []
        for _ in range(3):
            y = rng.randint(0, 2, size=50)
            p = np.clip(y + rng.normal(0, 0.3, size=50), 0.01, 0.99)
            folds_y.append(y)
            folds_p.append(p)
        return folds_y, folds_p

    def test_fold_specific_threshold(self, fold_data):
        folds_y, folds_p = fold_data
        result = FoldSpecificThreshold.optimize(
            folds_y, folds_p, youden_j, criterion_name="youden_j"
        )
        assert result.strategy == "fold_specific"
        assert isinstance(result.optimal_threshold, float)
        assert result.fold_thresholds is not None
        assert len(result.fold_thresholds) == 3

    def test_pooled_threshold(self, fold_data):
        folds_y, folds_p = fold_data
        result = PooledThreshold.optimize(folds_y, folds_p, youden_j, criterion_name="youden_j")
        assert result.strategy == "pooled"
        assert isinstance(result.optimal_threshold, float)
        assert result.fold_thresholds is None
