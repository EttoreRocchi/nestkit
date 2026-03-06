"""Tests for nestkit.inner.search -- build_search factory function."""

import unittest.mock

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from nestkit.inner.search import build_search


class TestBuildSearchGrid:
    def test_grid_returns_gridsearchcv(self, simple_classifier, simple_param_grid):
        result = build_search("grid", simple_classifier, simple_param_grid)
        assert isinstance(result, GridSearchCV)

    def test_grid_estimator_is_clone(self, simple_classifier, simple_param_grid):
        result = build_search("grid", simple_classifier, simple_param_grid)
        assert result.estimator is not simple_classifier

    def test_grid_param_grid_set(self, simple_classifier, simple_param_grid):
        result = build_search("grid", simple_classifier, simple_param_grid)
        assert result.param_grid == simple_param_grid

    def test_grid_scoring_and_cv_forwarded(self, simple_classifier, simple_param_grid):
        result = build_search(
            "grid", simple_classifier, simple_param_grid, cv=3, scoring="accuracy"
        )
        assert result.cv == 3
        assert result.scoring == "accuracy"

    def test_grid_verbose_reduced_by_two(self, simple_classifier, simple_param_grid):
        result = build_search("grid", simple_classifier, simple_param_grid, verbose=5)
        assert result.verbose == 3

    def test_grid_verbose_floor_at_zero(self, simple_classifier, simple_param_grid):
        result = build_search("grid", simple_classifier, simple_param_grid, verbose=0)
        assert result.verbose == 0


class TestBuildSearchRandom:
    def test_random_returns_randomizedsearchcv(self, simple_classifier, simple_param_grid):
        result = build_search("random", simple_classifier, simple_param_grid)
        assert isinstance(result, RandomizedSearchCV)

    def test_random_passes_random_state(self, simple_classifier, simple_param_grid):
        result = build_search("random", simple_classifier, simple_param_grid, random_state=42)
        assert result.random_state == 42

    def test_random_param_distributions_set(self, simple_classifier, simple_param_grid):
        result = build_search("random", simple_classifier, simple_param_grid)
        assert result.param_distributions == simple_param_grid


class TestBuildSearchBayesian:
    def test_bayesian_import_error_when_skopt_missing(self, simple_classifier, simple_param_grid):
        with (
            unittest.mock.patch.dict("sys.modules", {"skopt": None}),
            pytest.raises(ImportError, match="scikit-optimize"),
        ):
            build_search("bayesian", simple_classifier, simple_param_grid)


class TestBuildSearchDictScoring:
    def test_dict_scoring_refit_auto_set_to_first_key(self, simple_classifier, simple_param_grid):
        result = build_search(
            "grid",
            simple_classifier,
            simple_param_grid,
            scoring={"roc_auc": "roc_auc", "f1": "f1"},
            refit=True,
        )
        assert result.refit == "roc_auc"

    def test_dict_scoring_explicit_refit_preserved(self, simple_classifier, simple_param_grid):
        result = build_search(
            "grid",
            simple_classifier,
            simple_param_grid,
            scoring={"roc_auc": "roc_auc", "f1": "f1"},
            refit="f1",
        )
        assert result.refit == "f1"


class TestBuildSearchCustomInstance:
    def test_clone_existing_search_instance(self, simple_classifier, simple_param_grid):
        original = GridSearchCV(simple_classifier, simple_param_grid, cv=3)
        result = build_search(original, RandomForestClassifier(), simple_param_grid)
        assert isinstance(result, GridSearchCV)
        assert result is not original


class TestBuildSearchErrors:
    def test_invalid_string_strategy_raises_valueerror(self, simple_classifier, simple_param_grid):
        with pytest.raises(ValueError, match="search_strategy"):
            build_search("invalid", simple_classifier, simple_param_grid)

    @pytest.mark.parametrize("strategy", ["grid", "random"])
    def test_return_train_score_forwarded(self, strategy, simple_classifier, simple_param_grid):
        result = build_search(
            strategy, simple_classifier, simple_param_grid, return_train_score=True
        )
        assert result.return_train_score is True
