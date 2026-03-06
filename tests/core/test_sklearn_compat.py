"""Tests for scikit-learn estimator compatibility."""

from __future__ import annotations

import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

from nestkit.classifier import NestedCVClassifier


@pytest.fixture
def ncv():
    return NestedCVClassifier(
        estimator=RandomForestClassifier(random_state=0, n_estimators=5),
        param_grid={"max_depth": [2, 3]},
        outer_cv=2,
        inner_cv=2,
        return_estimator=True,
    )


def test_get_params(ncv):
    params = ncv.get_params(deep=False)
    assert isinstance(params, dict)
    expected_keys = {
        "estimator",
        "param_grid",
        "search_strategy",
        "outer_cv",
        "inner_cv",
        "scoring",
        "refit",
        "return_train_score",
        "return_estimator",
        "error_score",
        "n_jobs_outer",
        "n_jobs_inner",
        "verbose",
        "random_state",
        "callbacks",
        "pre_dispatch",
        "calibration_method",
        "threshold_strategy",
        "threshold_criterion",
        "threshold_beta",
        "cost_matrix",
        "min_recall",
        "calibration_cv",
    }
    assert expected_keys.issubset(params.keys())


def test_set_params(ncv):
    ncv.set_params(outer_cv=3, verbose=2)
    assert ncv.outer_cv == 3
    assert ncv.verbose == 2


def test_clone(ncv):
    cloned = clone(ncv)
    assert cloned is not ncv
    orig_params = ncv.get_params(deep=False)
    cloned_params = cloned.get_params(deep=False)
    for key in orig_params:
        if key == "estimator":
            assert type(orig_params[key]) is type(cloned_params[key])
            assert orig_params[key].get_params() == cloned_params[key].get_params()
        else:
            assert orig_params[key] == cloned_params[key], f"Mismatch on {key}"


def test_is_fitted(ncv, binary_data):
    assert not ncv.__sklearn_is_fitted__()

    X, y = binary_data
    ncv.fit(X, y)

    assert ncv.__sklearn_is_fitted__()
