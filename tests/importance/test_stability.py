"""Tests for nestkit.importance.stability -- nogueira_stability_index."""

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nestkit.importance.stability import nogueira_stability_index


class TestNogueiraStabilityIndex:
    def test_single_fold_returns_one(self):
        M = np.array([[0.5, 0.3, 0.2]])
        assert nogueira_stability_index(M, top_k=2) == 1.0

    def test_perfect_agreement(self):
        M = np.array([[0.9, 0.1, 0.05], [0.85, 0.15, 0.05], [0.95, 0.12, 0.03]])
        result = nogueira_stability_index(M, top_k=2)
        assert result == pytest.approx(1.0)

    def test_complete_disagreement(self):
        M = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        result = nogueira_stability_index(M, top_k=1)
        assert result < 0.5

    def test_top_k_equals_n_features(self):
        M = np.array([[0.5, 0.3, 0.2], [0.4, 0.35, 0.25]])
        result = nogueira_stability_index(M, top_k=3)
        assert result == 1.0

    def test_top_k_equals_one(self):
        M = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
        result = nogueira_stability_index(M, top_k=1)
        assert result == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "n_folds,n_features,top_k",
        [(2, 5, 2), (5, 10, 3), (10, 20, 5)],
    )
    def test_stability_in_valid_range(self, n_folds, n_features, top_k):
        rng = np.random.RandomState(42)
        M = rng.rand(n_folds, n_features)
        result = nogueira_stability_index(M, top_k=top_k)
        assert -1.0 <= result <= 1.0 + 1e-10

    def test_tied_importances_are_not_stability(self):
        """Constant importances give an arbitrary top-k, not a stable one.

        ``argsort`` breaks the ties by feature index, identically in every
        fold, so the index used to report a perfect 1.0 for a model that
        ranked nothing at all.
        """
        M = np.ones((5, 10))
        with pytest.warns(UserWarning, match="constant feature importances"):
            result = nogueira_stability_index(M, top_k=3)
        assert np.isnan(result)

    def test_tied_importances_reported_per_fold(self):
        """The warning names the offending folds."""
        M = np.array([[3.0, 2.0, 1.0], [1.0, 1.0, 1.0], [3.0, 1.0, 2.0]])
        with pytest.warns(UserWarning, match=r"Fold\(s\) \[1\]"):
            assert np.isnan(nogueira_stability_index(M, top_k=2))

    def test_constant_importances_fine_when_all_selected(self):
        """With top_k >= n_features every feature is selected regardless."""
        M = np.ones((4, 3))
        assert nogueira_stability_index(M, top_k=3) == 1.0

    def test_tied_importances_partial(self):
        """Some features tied, others distinct."""
        M = np.array(
            [
                [0.5, 0.5, 0.5, 0.1, 0.1],
                [0.5, 0.5, 0.5, 0.1, 0.1],
                [0.5, 0.5, 0.5, 0.1, 0.1],
            ]
        )
        result = nogueira_stability_index(M, top_k=3)
        assert -1.0 <= result <= 1.0 + 1e-10
        assert np.isfinite(result)
        # With identical data across folds, stability should be 1.0
        assert result == pytest.approx(1.0)

    @given(
        data=arrays(
            float,
            shape=st.tuples(st.integers(2, 8), st.integers(4, 15)),
            elements=st.floats(0, 1, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=30)
    def test_stability_bounded(self, data):
        top_k = min(3, data.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = nogueira_stability_index(data, top_k=top_k)
        # NaN is the documented answer when a fold has constant importances
        assert np.isnan(result) or -1.0 - 1e-10 <= result <= 1.0 + 1e-10
