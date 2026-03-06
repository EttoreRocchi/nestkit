"""Tests for NestedCVComparator (CRITICAL)."""

import numpy as np
import pytest

from nestkit.comparison.comparator import NestedCVComparator
from nestkit.comparison.statistical_tests import (
    bayesian_correlated_ttest,
    holm_bonferroni_correction,
    nadeau_bengio_corrected_ttest,
)
from nestkit.results.classifier_results import ClassifierOuterFoldResult, ClassifierResults


def _make_results(n_folds=5, test_indices_per_fold=None, scores=None, seed=0):
    """Build a ClassifierResults with mock fold data."""
    rng = np.random.RandomState(seed)
    results = ClassifierResults(n_outer_folds=n_folds)

    for i in range(n_folds):
        if test_indices_per_fold is not None:
            test_idx = test_indices_per_fold[i]
            n = len(test_idx)
        else:
            n = 20
            test_idx = np.arange(i * n, (i + 1) * n)

        y_true = rng.randint(0, 2, size=n)
        y_proba_raw = np.column_stack([1 - y_true * 0.8, y_true * 0.8])
        y_pred = (y_proba_raw[:, 1] >= 0.5).astype(int)

        score_val = scores[i] if scores is not None else 0.85 + rng.normal(0, 0.02)

        fr = ClassifierOuterFoldResult(
            fold_idx=i,
            train_indices=np.arange(200),
            test_indices=test_idx,
            best_params={"C": 1.0},
            best_inner_score=0.9,
            inner_cv_results={"mean_test_score": [0.9]},
            fit_time=0.1,
            score_time=0.01,
            fitted_estimator=None,
            y_true=y_true,
            y_proba_raw=y_proba_raw,
            y_pred_default=y_pred,
            outer_scores_default={"accuracy": score_val},
            confusion_matrix_default=np.array([[8, 2], [1, 9]]),
        )
        results.add_fold(fr)
    results.finalize()
    return results


class TestFoldAlignment:
    def test_fold_alignment_rejection(self):
        """Different test_indices across models should raise ValueError."""
        comp = NestedCVComparator()
        results_a = _make_results(
            n_folds=3,
            test_indices_per_fold=[np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])],
            seed=0,
        )
        results_b = _make_results(
            n_folds=3,
            test_indices_per_fold=[
                np.array([10, 11, 12]),
                np.array([13, 14, 15]),
                np.array([16, 17, 18]),
            ],
            seed=1,
        )
        comp.add("model_a", results_a)
        with pytest.raises(ValueError, match="test indices"):
            comp.add("model_b", results_b)

    def test_fold_alignment_acceptance(self):
        """Same test_indices across models should pass without error."""
        comp = NestedCVComparator()
        shared_indices = [np.arange(i * 20, (i + 1) * 20) for i in range(5)]
        results_a = _make_results(n_folds=5, test_indices_per_fold=shared_indices, seed=0)
        results_b = _make_results(n_folds=5, test_indices_per_fold=shared_indices, seed=1)
        comp.add("model_a", results_a)
        comp.add("model_b", results_b)  # Should not raise


class TestStatisticalTests:
    def test_identical_models_pvalue(self):
        """Identical scores should give p-value close to 1.0."""
        scores = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
        result = nadeau_bengio_corrected_ttest(scores, scores, n_train=80, n_test=20)
        assert result["p_value"] == pytest.approx(1.0)

    def test_corrected_vs_naive(self):
        """Corrected p-value should be >= naive paired t-test p-value."""
        rng = np.random.RandomState(42)
        scores_a = rng.normal(0.85, 0.02, size=10)
        scores_b = rng.normal(0.83, 0.02, size=10)

        corrected = nadeau_bengio_corrected_ttest(scores_a, scores_b, n_train=80, n_test=20)

        from scipy.stats import ttest_rel

        _, naive_p = ttest_rel(scores_a, scores_b)

        assert corrected["p_value"] >= naive_p - 1e-10

    def test_holm_correction(self):
        """Holm-corrected p-values should be >= uncorrected."""
        p_values = [0.01, 0.04, 0.03]
        corrected = holm_bonferroni_correction(p_values)
        for orig, corr in zip(p_values, corrected):
            assert corr >= orig - 1e-10

    def test_bayesian_comparison(self):
        """Bayesian test returns dict with required probability keys."""
        scores_a = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
        scores_b = np.array([0.80, 0.82, 0.78, 0.81, 0.79])
        result = bayesian_correlated_ttest(scores_a, scores_b, n_train=80, n_test=20, rope=0.01)
        assert isinstance(result, dict)
        assert "p_a_better" in result
        assert "p_equivalent" in result
        assert "p_b_better" in result
        total = result["p_a_better"] + result["p_equivalent"] + result["p_b_better"]
        assert total == pytest.approx(1.0, abs=1e-6)


class TestComparatorSummary:
    @pytest.fixture
    def two_model_comparator(self):
        shared_indices = [np.arange(i * 20, (i + 1) * 20) for i in range(5)]
        comp = NestedCVComparator()
        comp.add(
            "model_a",
            _make_results(
                n_folds=5,
                test_indices_per_fold=shared_indices,
                scores=[0.90, 0.91, 0.89, 0.92, 0.88],
                seed=0,
            ),
        )
        comp.add(
            "model_b",
            _make_results(
                n_folds=5,
                test_indices_per_fold=shared_indices,
                scores=[0.80, 0.81, 0.79, 0.82, 0.78],
                seed=1,
            ),
        )
        return comp

    def test_summary_returns_correct_columns(self, two_model_comparator):
        df = two_model_comparator.summary("accuracy")
        expected_cols = {
            "model",
            "mean",
            "std",
            "median",
            "ci_lower",
            "ci_upper",
            "min",
            "max",
            "iqr",
        }
        assert set(df.columns) == expected_cols

    def test_summary_correct_row_count(self, two_model_comparator):
        df = two_model_comparator.summary("accuracy")
        assert len(df) == 2

    def test_summary_ci_lower_lt_mean_lt_ci_upper(self, two_model_comparator):
        df = two_model_comparator.summary("accuracy")
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["mean"] <= row["ci_upper"]


class TestRankModels:
    @pytest.fixture
    def two_model_comparator(self):
        shared_indices = [np.arange(i * 20, (i + 1) * 20) for i in range(5)]
        comp = NestedCVComparator()
        comp.add(
            "model_a",
            _make_results(
                n_folds=5,
                test_indices_per_fold=shared_indices,
                scores=[0.90, 0.91, 0.89, 0.92, 0.88],
                seed=0,
            ),
        )
        comp.add(
            "model_b",
            _make_results(
                n_folds=5,
                test_indices_per_fold=shared_indices,
                scores=[0.80, 0.81, 0.79, 0.82, 0.78],
                seed=1,
            ),
        )
        return comp

    def test_rank_sorted_descending(self, two_model_comparator):
        df = two_model_comparator.rank_models("accuracy")
        assert df["mean"].iloc[0] >= df["mean"].iloc[1]

    def test_rank_starts_at_one(self, two_model_comparator):
        df = two_model_comparator.rank_models("accuracy")
        assert list(df["rank"]) == [1, 2]

    def test_best_model_first(self, two_model_comparator):
        df = two_model_comparator.rank_models("accuracy")
        assert df["model"].iloc[0] == "model_a"


class TestGetScores:
    def test_get_scores_default(self):
        comp = NestedCVComparator()
        results = _make_results(n_folds=3, scores=[0.85, 0.87, 0.83], seed=0)
        comp.add("m", results)
        scores = comp._get_scores("m", "accuracy")
        assert len(scores) == 3

    def test_get_scores_optimized_raises_when_missing(self):
        comp = NestedCVComparator()
        results = _make_results(n_folds=3, seed=0)
        comp.add("m", results)
        with pytest.raises(ValueError, match="no optimized scores"):
            comp._get_scores("m", "accuracy", threshold="optimized")


class TestPairwiseCorrectedTtest:
    def test_three_models_three_pairs(self):
        shared_indices = [np.arange(i * 20, (i + 1) * 20) for i in range(5)]
        comp = NestedCVComparator()
        for name, seed in [("a", 0), ("b", 1), ("c", 2)]:
            comp.add(
                name, _make_results(n_folds=5, test_indices_per_fold=shared_indices, seed=seed)
            )
        df = comp.pairwise_corrected_ttest("accuracy")
        assert len(df) == 3
        assert "p_value_corrected" in df.columns

    def test_corrected_pvalue_geq_uncorrected(self):
        shared_indices = [np.arange(i * 20, (i + 1) * 20) for i in range(5)]
        comp = NestedCVComparator()
        comp.add("a", _make_results(n_folds=5, test_indices_per_fold=shared_indices, seed=0))
        comp.add("b", _make_results(n_folds=5, test_indices_per_fold=shared_indices, seed=1))
        df = comp.pairwise_corrected_ttest("accuracy")
        for _, row in df.iterrows():
            assert row["p_value_corrected"] >= row["p_value"] - 1e-10


class TestBayesianComparison:
    def test_bayesian_comparison_returns_keys(self):
        shared_indices = [np.arange(i * 20, (i + 1) * 20) for i in range(5)]
        comp = NestedCVComparator()
        comp.add("a", _make_results(n_folds=5, test_indices_per_fold=shared_indices, seed=0))
        comp.add("b", _make_results(n_folds=5, test_indices_per_fold=shared_indices, seed=1))
        result = comp.bayesian_comparison("accuracy", "a", "b")
        assert "p_a_better" in result
        assert "p_equivalent" in result
        assert "p_b_better" in result
