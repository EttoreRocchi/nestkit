"""Tests for nestkit.comparison.statistical_tests.

Covers: nadeau_bengio_corrected_ttest, bayesian_correlated_ttest,
        holm_bonferroni_correction.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nestkit.comparison.statistical_tests import (
    bayesian_correlated_ttest,
    holm_bonferroni_correction,
    nadeau_bengio_corrected_ttest,
)


class TestNadeauBengioCorrectedTtest:
    def test_identical_scores_pvalue_one(self):
        scores = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
        result = nadeau_bengio_corrected_ttest(scores, scores, 80, 20)
        assert result["p_value"] == pytest.approx(1.0)
        assert result["t_statistic"] == 0.0

    def test_identical_scores_not_significant(self):
        scores = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
        result = nadeau_bengio_corrected_ttest(scores, scores, 80, 20)
        assert result["significant_at_005"] is False
        assert result["significant_at_001"] is False

    def test_clearly_different_scores_significant(self):
        a = np.array([0.95, 0.94, 0.96, 0.95, 0.94])
        b = np.array([0.50, 0.51, 0.49, 0.50, 0.51])
        result = nadeau_bengio_corrected_ttest(a, b, 160, 40)
        assert result["significant_at_005"] is True
        assert result["p_value"] < 0.05

    def test_mean_difference_sign(self):
        a = np.array([0.9, 0.91, 0.89])
        b = np.array([0.8, 0.81, 0.79])
        result = nadeau_bengio_corrected_ttest(a, b, 80, 20)
        assert result["mean_difference"] > 0

    def test_ci_contains_mean_difference(self):
        a = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
        b = np.array([0.80, 0.82, 0.78, 0.81, 0.79])
        result = nadeau_bengio_corrected_ttest(a, b, 80, 20)
        assert result["ci_lower"] <= result["mean_difference"] <= result["ci_upper"]

    def test_return_keys_complete(self):
        a = np.array([0.85, 0.87, 0.83])
        b = np.array([0.80, 0.82, 0.78])
        result = nadeau_bengio_corrected_ttest(a, b, 80, 20)
        expected_keys = {
            "t_statistic",
            "p_value",
            "mean_difference",
            "corrected_std",
            "ci_lower",
            "ci_upper",
            "n_folds",
            "significant_at_005",
            "significant_at_001",
        }
        assert set(result.keys()) == expected_keys

    def test_two_folds_minimum(self):
        a = np.array([0.9, 0.8])
        b = np.array([0.7, 0.6])
        result = nadeau_bengio_corrected_ttest(a, b, 80, 20)
        assert result["n_folds"] == 2
        assert np.isfinite(result["p_value"])

    @given(
        a=arrays(float, shape=5, elements=st.floats(0, 1, allow_nan=False, allow_infinity=False)),
        b=arrays(float, shape=5, elements=st.floats(0, 1, allow_nan=False, allow_infinity=False)),
    )
    @settings(max_examples=50)
    def test_p_value_in_zero_one(self, a, b):
        result = nadeau_bengio_corrected_ttest(a, b, 80, 20)
        assert 0.0 <= result["p_value"] <= 1.0


class TestBayesianCorrelatedTtest:
    def test_probabilities_sum_to_one(self):
        a = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
        b = np.array([0.80, 0.82, 0.78, 0.81, 0.79])
        result = bayesian_correlated_ttest(a, b, 80, 20, rope=0.01)
        total = result["p_a_better"] + result["p_equivalent"] + result["p_b_better"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_a_clearly_better(self):
        a = np.array([0.95, 0.94, 0.96, 0.95, 0.94])
        b = np.array([0.50, 0.51, 0.49, 0.50, 0.51])
        result = bayesian_correlated_ttest(a, b, 160, 40, rope=0.01)
        assert result["p_a_better"] > 0.9

    def test_b_clearly_better(self):
        a = np.array([0.50, 0.51, 0.49, 0.50, 0.51])
        b = np.array([0.95, 0.94, 0.96, 0.95, 0.94])
        result = bayesian_correlated_ttest(a, b, 160, 40, rope=0.01)
        assert result["p_b_better"] > 0.9

    def test_equivalent_scores_high_p_equivalent(self):
        a = np.array([0.850, 0.851, 0.849, 0.850, 0.850])
        b = np.array([0.850, 0.849, 0.851, 0.850, 0.850])
        result = bayesian_correlated_ttest(a, b, 160, 40, rope=0.01)
        assert result["p_equivalent"] > 0.5

    def test_zero_std_within_rope(self):
        a = np.array([0.85, 0.85, 0.85])
        result = bayesian_correlated_ttest(a, a, 80, 20, rope=0.01)
        assert result["p_equivalent"] == 1.0

    def test_zero_std_outside_rope(self):
        a = np.array([0.90, 0.90, 0.90])
        b = np.array([0.80, 0.80, 0.80])
        result = bayesian_correlated_ttest(a, b, 80, 20, rope=0.01)
        assert result["p_a_better"] == 1.0
        assert result["p_b_better"] == 0.0

    def test_return_keys_complete(self):
        a = np.array([0.85, 0.87, 0.83])
        b = np.array([0.80, 0.82, 0.78])
        result = bayesian_correlated_ttest(a, b, 80, 20)
        expected_keys = {
            "p_a_better",
            "p_equivalent",
            "p_b_better",
            "rope",
            "mean_difference",
            "hdi_lower",
            "hdi_upper",
        }
        assert set(result.keys()) == expected_keys

    def test_hdi_contains_mean_difference(self):
        a = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
        b = np.array([0.80, 0.82, 0.78, 0.81, 0.79])
        result = bayesian_correlated_ttest(a, b, 80, 20)
        assert result["hdi_lower"] <= result["mean_difference"] <= result["hdi_upper"]

    @given(
        a=arrays(float, shape=5, elements=st.floats(0.01, 0.99, allow_nan=False)),
        b=arrays(float, shape=5, elements=st.floats(0.01, 0.99, allow_nan=False)),
        rope=st.floats(0.0, 0.5),
    )
    @settings(max_examples=50)
    def test_probabilities_nonnegative_and_sum_one(self, a, b, rope):
        result = bayesian_correlated_ttest(a, b, 80, 20, rope)
        assert result["p_a_better"] >= 0
        assert result["p_equivalent"] >= 0
        assert result["p_b_better"] >= 0
        total = result["p_a_better"] + result["p_equivalent"] + result["p_b_better"]
        assert total == pytest.approx(1.0, abs=1e-4)


class TestHolmBonferroniCorrection:
    def test_known_example(self):
        result = holm_bonferroni_correction([0.01, 0.04, 0.03])
        assert result == pytest.approx([0.03, 0.06, 0.06], abs=1e-10)

    def test_single_pvalue_unchanged(self):
        result = holm_bonferroni_correction([0.03])
        assert result == pytest.approx([0.03])

    def test_all_one_remain_one(self):
        result = holm_bonferroni_correction([1.0, 1.0, 1.0])
        assert result == pytest.approx([1.0, 1.0, 1.0])

    def test_clipped_to_one(self):
        result = holm_bonferroni_correction([0.5, 0.6, 0.7])
        for val in result:
            assert val <= 1.0

    def test_monotonicity_enforced(self):
        pvals = [0.01, 0.02, 0.03, 0.04, 0.05]
        result = holm_bonferroni_correction(pvals)
        sorted_idx = np.argsort(pvals)
        for i in range(1, len(sorted_idx)):
            assert result[sorted_idx[i]] >= result[sorted_idx[i - 1]]

    def test_empty_list(self):
        result = holm_bonferroni_correction([])
        assert result == []

    @given(st.lists(st.floats(0, 1), min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_corrected_geq_original(self, pvals):
        corrected = holm_bonferroni_correction(pvals)
        for orig, corr in zip(pvals, corrected):
            assert corr >= orig - 1e-10

    @given(st.lists(st.floats(0, 1), min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_corrected_leq_one(self, pvals):
        corrected = holm_bonferroni_correction(pvals)
        for val in corrected:
            assert val <= 1.0 + 1e-10
