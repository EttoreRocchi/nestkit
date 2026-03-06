"""Edge-case tests for NestedCVClassifier."""

from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from nestkit.classifier import NestedCVClassifier


def test_single_param_config():
    """param_grid with a single value should work without error."""
    X, y = make_classification(n_samples=60, n_features=5, random_state=0)
    ncv = NestedCVClassifier(
        estimator=RandomForestClassifier(random_state=0, n_estimators=5),
        param_grid={"max_depth": [3]},
        outer_cv=2,
        inner_cv=2,
        return_estimator=True,
    )
    ncv.fit(X, y)
    assert ncv.is_fitted_
    assert len(ncv.results_.fold_results_) == 2


def test_minimal_folds():
    """outer_cv=2, inner_cv=2  -  smallest valid fold counts."""
    X, y = make_classification(n_samples=60, n_features=5, random_state=0)
    ncv = NestedCVClassifier(
        estimator=RandomForestClassifier(random_state=0, n_estimators=5),
        param_grid={"max_depth": [2, 4]},
        outer_cv=2,
        inner_cv=2,
        return_estimator=True,
    )
    ncv.fit(X, y)
    assert ncv.results_.n_outer_folds_ == 2
    assert len(ncv.results_.fold_results_) == 2
    assert not ncv.results_.outer_scores_default_.empty
