"""Tests for diagnostics module -- HyperparameterStability."""

import numpy as np
import pandas as pd
import pytest

from nestkit.diagnostics.stability import HyperparameterStability


class TestHyperparameterStability:
    @pytest.fixture
    def stability(self):
        params = [
            {"C": 1.0, "kernel": "rbf"},
            {"C": 1.0, "kernel": "rbf"},
            {"C": 10.0, "kernel": "rbf"},
        ]
        return HyperparameterStability(params)

    def test_stability_summary(self, stability):
        df = stability.summary()
        assert isinstance(df, pd.DataFrame)
        assert "param" in df.columns
        assert "mode" in df.columns
        assert "agreement_rate" in df.columns
        assert "entropy" in df.columns

    def test_is_stable(self, stability):
        result = stability.is_stable(threshold=0.5)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, bool)
        assert result["kernel"] is True

    def test_pairwise_jaccard(self, stability):
        df = stability.pairwise_jaccard()
        assert isinstance(df, pd.DataFrame)
        assert "fold_i" in df.columns
        assert "fold_j" in df.columns
        assert "jaccard" in df.columns
        assert len(df) == 3


class TestHyperparameterStabilityEdgeCases:
    def test_single_fold(self):
        hs = HyperparameterStability([{"C": 1.0}])
        df = hs.pairwise_jaccard()
        assert len(df) == 0
        summary = hs.summary()
        assert len(summary) == 1

    def test_all_identical_params(self):
        params = [{"C": 1.0, "k": "rbf"}] * 5
        hs = HyperparameterStability(params)
        summary = hs.summary()
        for _, row in summary.iterrows():
            assert row["agreement_rate"] == 1.0
            assert row["entropy"] == pytest.approx(0.0, abs=1e-10)
        jac = hs.pairwise_jaccard()
        assert all(jac["jaccard"] == 1.0)

    def test_all_different_params(self):
        params = [{"C": i} for i in range(3)]
        hs = HyperparameterStability(params)
        summary = hs.summary()
        assert summary.iloc[0]["agreement_rate"] == pytest.approx(1 / 3)

    def test_missing_param_in_some_folds(self):
        params = [{"C": 1.0}, {"C": 1.0, "gamma": 0.1}]
        hs = HyperparameterStability(params)
        summary = hs.summary()
        assert "gamma" in summary["param"].values
        assert "C" in summary["param"].values

    def test_non_numeric_params_cv_is_nan(self):
        params = [{"kernel": "rbf"}, {"kernel": "linear"}]
        hs = HyperparameterStability(params)
        summary = hs.summary()
        row = summary[summary["param"] == "kernel"].iloc[0]
        assert np.isnan(row["cv"])

    def test_is_stable_threshold_boundary(self):
        params = [{"C": 1.0}, {"C": 1.0}, {"C": 2.0}]
        hs = HyperparameterStability(params)
        result = hs.is_stable(threshold=2 / 3)
        assert result["C"] is True

    def test_is_stable_threshold_one(self):
        params = [{"C": 1.0}, {"C": 2.0}]
        hs = HyperparameterStability(params)
        result = hs.is_stable(threshold=1.0)
        assert result["C"] is False


class TestPairwiseJaccard:
    def test_identical_configs_jaccard_one(self):
        params = [{"a": 1, "b": 2}] * 3
        hs = HyperparameterStability(params)
        jac = hs.pairwise_jaccard()
        assert all(jac["jaccard"] == 1.0)

    def test_completely_different_configs_jaccard_zero(self):
        params = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        hs = HyperparameterStability(params)
        jac = hs.pairwise_jaccard()
        assert jac["jaccard"].iloc[0] == 0.0

    def test_partial_overlap_jaccard(self):
        params = [{"a": 1, "b": 2}, {"a": 1, "b": 3}]
        hs = HyperparameterStability(params)
        jac = hs.pairwise_jaccard()
        assert jac["jaccard"].iloc[0] == pytest.approx(1 / 3)

    def test_empty_params_jaccard_one(self):
        params = [{}, {}]
        hs = HyperparameterStability(params)
        jac = hs.pairwise_jaccard()
        assert jac["jaccard"].iloc[0] == 1.0

    def test_two_folds_one_pair(self):
        params = [{"C": 1.0}, {"C": 2.0}]
        hs = HyperparameterStability(params)
        jac = hs.pairwise_jaccard()
        assert len(jac) == 1
