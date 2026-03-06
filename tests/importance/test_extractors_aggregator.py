"""Tests for feature importance extraction, aggregation, and stability."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nestkit.importance.aggregator import FeatureImportanceAggregator
from nestkit.importance.extractors import extract_model_importance
from nestkit.importance.stability import nogueira_stability_index


@dataclass
class _FakeFoldResult:
    fitted_estimator: object
    test_indices: np.ndarray = field(default_factory=lambda: np.arange(10))


class _FakeResults:
    """Minimal stand-in for _BaseNestedCVResults."""

    def __init__(self, estimators, feature_names=None):
        self.fold_results_ = [_FakeFoldResult(fitted_estimator=est) for est in estimators]
        self.feature_names_in_ = feature_names or [f"feature_{i}" for i in range(4)]

    @property
    def has_fitted_estimators(self):
        return all(fr.fitted_estimator is not None for fr in self.fold_results_)


def _make_fitted_rf(importances):
    """Return a mock estimator with known feature_importances_."""
    est = MagicMock()
    est.feature_importances_ = np.array(importances, dtype=float)
    del est.steps
    return est


def test_model_importance_rf():
    importances = [0.4, 0.3, 0.2, 0.1]
    est = _make_fitted_rf(importances)
    result = extract_model_importance(est)
    np.testing.assert_array_almost_equal(result, importances)


def test_pipeline_unwrapping():
    rf = RandomForestClassifier(n_estimators=5, random_state=0)
    X = np.random.RandomState(0).randn(30, 4)
    y = np.array([0, 1, 0] * 10)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", rf)])
    pipe.fit(X, y)

    imp = extract_model_importance(pipe)
    assert imp.shape == (4,)
    np.testing.assert_array_almost_equal(imp, rf.feature_importances_)


def test_normalize():
    importances_a = [0.5, 0.3, 0.15, 0.05]
    importances_b = [0.2, 0.4, 0.1, 0.3]
    estimators = [_make_fitted_rf(importances_a), _make_fitted_rf(importances_b)]
    results = _FakeResults(estimators)

    agg = FeatureImportanceAggregator(results, method="model", normalize=True)
    agg.compute()

    for fold_imp in agg.importances_per_fold_:
        assert abs(fold_imp.sum() - 1.0) < 1e-9, "Per-fold sum should be ~1.0"


def test_no_fit_called():
    """compute() must not call .fit() on the stored estimators."""
    est = _make_fitted_rf([0.25, 0.25, 0.25, 0.25])
    est.fit = MagicMock()
    results = _FakeResults([est, est])

    agg = FeatureImportanceAggregator(results, method="model", normalize=False)
    agg.compute()

    est.fit.assert_not_called()


def test_missing_estimators():
    results = _FakeResults([None, None])
    with pytest.raises(ValueError, match="return_estimator=True"):
        FeatureImportanceAggregator(results, method="model")


def test_stability_index_identical():
    matrix = np.array(
        [
            [0.5, 0.3, 0.15, 0.05],
            [0.5, 0.3, 0.15, 0.05],
            [0.5, 0.3, 0.15, 0.05],
        ]
    )
    idx = nogueira_stability_index(matrix, top_k=2)
    assert idx == pytest.approx(1.0)


def test_consensus_features():
    importances_a = [0.5, 0.3, 0.15, 0.05]
    importances_b = [0.45, 0.35, 0.1, 0.1]
    estimators = [_make_fitted_rf(importances_a), _make_fitted_rf(importances_b)]
    results = _FakeResults(estimators, feature_names=["a", "b", "c", "d"])

    agg = FeatureImportanceAggregator(results, method="model", normalize=False)
    agg.compute()

    consensus = agg.consensus_features(criterion="top_k", top_k=2)
    assert isinstance(consensus, list)
    assert all(isinstance(f, str) for f in consensus)
    assert len(consensus) == 2


def test_pairwise_rank_correlation():
    importances_a = [0.5, 0.3, 0.15, 0.05]
    importances_b = [0.45, 0.35, 0.1, 0.1]
    importances_c = [0.6, 0.2, 0.15, 0.05]
    estimators = [
        _make_fitted_rf(importances_a),
        _make_fitted_rf(importances_b),
        _make_fitted_rf(importances_c),
    ]
    results = _FakeResults(estimators)

    agg = FeatureImportanceAggregator(results, method="model", normalize=False)
    agg.compute()

    df = agg.pairwise_rank_correlation()
    assert isinstance(df, pd.DataFrame)
    assert "spearman_r" in df.columns
    assert len(df) == 3
