"""Tests for base class mechanics (_BaseNestedCV via concrete subclasses)."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from nestkit import NestedCVClassifier


class TestGetSetParams:
    """Test sklearn-compatible get_params, set_params, and clone."""

    def test_get_set_params(self, simple_classifier, simple_param_grid):
        ncv = NestedCVClassifier(
            estimator=simple_classifier,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
        )

        params = ncv.get_params()
        assert "outer_cv" in params
        assert "inner_cv" in params
        assert "estimator" in params
        assert "param_grid" in params
        assert params["outer_cv"] == 3
        assert params["inner_cv"] == 2

        result = ncv.set_params(outer_cv=5)
        assert result is ncv
        assert ncv.outer_cv == 5
        assert ncv.get_params()["outer_cv"] == 5

        ncv2 = clone(ncv)
        assert ncv2 is not ncv
        assert ncv2.get_params()["outer_cv"] == 5
        assert ncv2.get_params()["inner_cv"] == 2


class TestSklearnIsFitted:
    """Test __sklearn_is_fitted__ before and after fit."""

    def test_sklearn_is_fitted(self, binary_data, simple_param_grid, simple_classifier):
        X, y = binary_data
        ncv = NestedCVClassifier(
            estimator=simple_classifier,
            param_grid=simple_param_grid,
            outer_cv=3,
            inner_cv=2,
        )

        assert ncv.__sklearn_is_fitted__() is False
        with pytest.raises(NotFittedError):
            check_is_fitted(ncv)

        ncv.fit(X, y)
        assert ncv.__sklearn_is_fitted__() is True
        check_is_fitted(ncv)


class TestGroupsPropagation:
    """Test that groups are passed through to the inner CV search."""

    @pytest.mark.filterwarnings("ignore:The groups parameter is ignored by StratifiedKFold")
    def test_groups_propagation(self, binary_data, simple_param_grid):
        X, y = binary_data
        n_samples = X.shape[0]

        groups = np.array([i % 10 for i in range(n_samples)])

        from sklearn.model_selection import GroupKFold

        ncv = NestedCVClassifier(
            estimator=RandomForestClassifier(random_state=42, n_estimators=10),
            param_grid=simple_param_grid,
            outer_cv=GroupKFold(n_splits=3),
            inner_cv=2,
        )

        ncv.fit(X, y, groups=groups)
        assert ncv.is_fitted_ is True

        results = ncv.results_
        assert len(results.best_params_per_fold_) == 3

        for i, fr_i in enumerate(results.fold_results_):
            for j, fr_j in enumerate(results.fold_results_):
                if i >= j:
                    continue
                groups_i = set(groups[fr_i.test_indices])
                groups_j = set(groups[fr_j.test_indices])
                assert groups_i.isdisjoint(groups_j), (
                    f"Folds {i} and {j} share groups: {groups_i & groups_j}"
                )
