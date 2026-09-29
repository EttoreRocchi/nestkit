"""Tests for CoxPHWrapper sklearn compatibility."""

import numpy as np
import pytest
from sklearn.base import clone

from nestkit.survival import CoxPHWrapper

pytest.importorskip("lifelines")


@pytest.fixture
def fitted_wrapper(survival_data):
    X, y = survival_data
    wrapper = CoxPHWrapper(penalizer=0.1)
    wrapper.fit(X, y)
    return wrapper, X, y


class TestCoxPHWrapperFit:
    def test_fit_returns_self(self, survival_data):
        X, y = survival_data
        wrapper = CoxPHWrapper(penalizer=0.1)
        result = wrapper.fit(X, y)
        assert result is wrapper

    def test_fit_stores_fitter(self, fitted_wrapper):
        wrapper, _, _ = fitted_wrapper
        assert hasattr(wrapper, "fitter_")

    def test_feature_names_stored(self, fitted_wrapper):
        wrapper, X, _ = fitted_wrapper
        assert len(wrapper.feature_names_in_) == X.shape[1]
        assert wrapper.n_features_in_ == X.shape[1]


class TestCoxPHWrapperPredict:
    def test_predict_shape(self, fitted_wrapper):
        wrapper, X, _ = fitted_wrapper
        risk_scores = wrapper.predict(X)
        assert risk_scores.shape == (X.shape[0],)

    def test_predict_is_numeric(self, fitted_wrapper):
        wrapper, X, _ = fitted_wrapper
        risk_scores = wrapper.predict(X)
        assert np.isfinite(risk_scores).all()


class TestCoxPHWrapperScore:
    def test_score_in_range(self, fitted_wrapper):
        wrapper, X, y = fitted_wrapper
        score = wrapper.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_score_reasonable(self, fitted_wrapper):
        """Score should be better than random (0.5) on this synthetic data."""
        wrapper, X, y = fitted_wrapper
        score = wrapper.score(X, y)
        assert score > 0.5


class TestCoxPHWrapperSklearn:
    def test_get_params(self):
        wrapper = CoxPHWrapper(penalizer=0.5, l1_ratio=0.3)
        params = wrapper.get_params()
        assert params["penalizer"] == 0.5
        assert params["l1_ratio"] == 0.3

    def test_set_params(self):
        wrapper = CoxPHWrapper()
        wrapper.set_params(penalizer=1.0)
        assert wrapper.penalizer == 1.0

    def test_clone(self):
        wrapper = CoxPHWrapper(penalizer=0.5, l1_ratio=0.3)
        cloned = clone(wrapper)
        assert cloned.penalizer == 0.5
        assert cloned.l1_ratio == 0.3
        assert cloned is not wrapper

    def test_clone_unfitted(self, fitted_wrapper):
        wrapper, _, _ = fitted_wrapper
        cloned = clone(wrapper)
        assert not hasattr(cloned, "fitter_")


class TestCoxPHWrapperExtraMethods:
    def test_predict_survival_function(self, fitted_wrapper):
        wrapper, X, _ = fitted_wrapper
        sf = wrapper.predict_survival_function(X[:5])
        assert sf.shape[1] == 5
        # Survival probabilities should be in [0, 1]
        assert (sf.values >= 0).all()
        assert (sf.values <= 1).all()

    def test_predict_survival_function_at_times(self, fitted_wrapper):
        wrapper, X, _ = fitted_wrapper
        times = np.array([1.0, 5.0, 10.0])
        sf = wrapper.predict_survival_function(X[:5], times=times)
        assert sf.shape == (3, 5)

    def test_predict_median(self, fitted_wrapper):
        wrapper, X, _ = fitted_wrapper
        medians = wrapper.predict_median(X[:5])
        assert medians.shape == (5,)
