"""Tests for InnerCVReport."""

import pandas as pd
import pytest

from nestkit.inner.tuning_report import InnerCVReport


@pytest.fixture
def cv_results():
    return {
        "mean_test_score": [0.80, 0.85, 0.90, 0.88, 0.82],
        "std_test_score": [0.02, 0.03, 0.01, 0.02, 0.04],
        "rank_test_score": [5, 3, 1, 2, 4],
        "param_C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "param_kernel": ["rbf", "rbf", "rbf", "linear", "linear"],
    }


@pytest.fixture
def report(cv_results):
    return InnerCVReport(cv_results, outer_fold_idx=0)


class TestInnerCVReport:
    def test_to_dataframe(self, report):
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "mean_test_score" in df.columns

    def test_ranking(self, report):
        df = report.ranking()
        assert isinstance(df, pd.DataFrame)
        assert df["rank_test_score"].iloc[0] == 1

    def test_top_k(self, report):
        k = 3
        df = report.top_k(k=k)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == k

    def test_param_importance(self, report):
        df = report.param_importance()
        assert isinstance(df, pd.DataFrame)
        assert "parameter" in df.columns
        assert len(df) > 0


class TestScoreDistribution:
    def test_score_distribution_returns_sorted_dataframe(self, report):
        df = report.score_distribution("C")
        assert isinstance(df, pd.DataFrame)
        assert df["param_C"].is_monotonic_increasing

    def test_score_distribution_includes_std_column(self, report):
        df = report.score_distribution("C")
        assert "std_test_score" in df.columns

    def test_score_distribution_missing_param_returns_empty(self, report):
        df = report.score_distribution("nonexistent")
        assert len(df) == 0

    def test_score_distribution_missing_score_returns_empty(self):
        cv_results = {"param_C": [0.1, 1.0]}
        r = InnerCVReport(cv_results, outer_fold_idx=0)
        df = r.score_distribution("C")
        assert len(df) == 0

    def test_score_distribution_with_named_metric(self):
        cv_results = {
            "mean_test_roc_auc": [0.80, 0.85, 0.90],
            "std_test_roc_auc": [0.02, 0.01, 0.03],
            "param_C": [0.1, 1.0, 10.0],
        }
        r = InnerCVReport(cv_results, outer_fold_idx=0)
        df = r.score_distribution("C", metric="roc_auc")
        assert len(df) == 3
        assert "mean_test_roc_auc" in df.columns


class TestInnerCVReportEdgeCases:
    def test_ranking_without_rank_column_falls_back_to_score(self):
        cv_results = {"mean_test_score": [0.80, 0.90, 0.85], "param_C": [0.1, 1.0, 10.0]}
        r = InnerCVReport(cv_results, outer_fold_idx=0)
        df = r.ranking()
        assert df["mean_test_score"].iloc[0] == 0.90

    def test_ranking_without_score_column_returns_df_as_is(self):
        cv_results = {"param_C": [0.1, 1.0]}
        r = InnerCVReport(cv_results, outer_fold_idx=0)
        df = r.ranking()
        assert len(df) == 2

    def test_top_k_larger_than_configs_returns_all(self, report):
        df = report.top_k(k=100)
        assert len(df) == 5

    def test_param_importance_no_score_column_returns_empty(self):
        cv_results = {"param_C": [0.1, 1.0]}
        r = InnerCVReport(cv_results, outer_fold_idx=0)
        df = r.param_importance()
        assert len(df) == 0

    def test_param_importance_single_value_per_param(self):
        cv_results = {
            "mean_test_score": [0.85],
            "param_C": [1.0],
        }
        r = InnerCVReport(cv_results, outer_fold_idx=0)
        df = r.param_importance()
        assert df["variance_explained"].iloc[0] == 0.0

    def test_param_importance_relative_importance_sums_to_one(self, report):
        df = report.param_importance()
        if not df.empty:
            total = df["relative_importance"].sum()
            assert total == pytest.approx(1.0, abs=1e-10)

    def test_ranking_with_named_metric(self):
        cv_results = {
            "mean_test_roc_auc": [0.80, 0.90],
            "rank_test_roc_auc": [2, 1],
            "param_C": [0.1, 1.0],
        }
        r = InnerCVReport(cv_results, outer_fold_idx=0)
        df = r.ranking(metric="roc_auc")
        assert df["rank_test_roc_auc"].iloc[0] == 1
