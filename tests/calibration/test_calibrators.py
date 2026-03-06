"""Tests for calibration module."""

import numpy as np
import pandas as pd
import pytest

from nestkit.calibration.calibrators import PostHocCalibrator
from nestkit.calibration.diagnostics import CalibrationDiagnostics


@pytest.fixture
def binary_data():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=200)
    y_proba = np.clip(y_true + rng.normal(0, 0.3, size=200), 0.01, 0.99)
    return y_true, y_proba


class TestPostHocCalibrator:
    def test_sigmoid_calibrator(self, binary_data):
        y_true, y_proba = binary_data
        cal = PostHocCalibrator("sigmoid")
        cal.fit(y_proba, y_true)
        result = cal.predict_proba(y_proba)
        assert result.ndim == 2
        assert result.shape == (len(y_proba), 2)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)

    def test_isotonic_calibrator(self, binary_data):
        y_true, y_proba = binary_data
        cal = PostHocCalibrator("isotonic")
        cal.fit(y_proba, y_true)
        result = cal.predict_proba(y_proba)
        assert result.ndim == 2
        assert result.shape == (len(y_proba), 2)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)

    def test_beta_calibrator(self, binary_data):
        y_true, y_proba = binary_data
        cal = PostHocCalibrator("beta")
        cal.fit(y_proba, y_true)
        result = cal.predict_proba(y_proba)
        assert result.ndim == 2
        assert result.shape == (len(y_proba), 2)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)

    def test_venn_abers_calibrator(self, binary_data):
        y_true, y_proba = binary_data
        y_true_small = y_true[:50]
        y_proba_small = y_proba[:50]
        cal = PostHocCalibrator("venn_abers")
        cal.fit(y_proba_small, y_true_small)
        result = cal.predict_proba(y_proba_small[:10])
        assert result.ndim == 2
        assert result.shape == (10, 2)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)


class TestCalibrationDiagnostics:
    def test_ece(self, binary_data):
        y_true, y_proba = binary_data
        ece = CalibrationDiagnostics.expected_calibration_error(y_true, y_proba)
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0

    def test_ece_uniform(self, binary_data):
        y_true, y_proba = binary_data
        ece = CalibrationDiagnostics.expected_calibration_error(
            y_true, y_proba, strategy="uniform"
        )
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0

    def test_ece_quantile_vs_uniform(self, binary_data):
        y_true, y_proba = binary_data
        ece_q = CalibrationDiagnostics.expected_calibration_error(
            y_true, y_proba, strategy="quantile"
        )
        ece_u = CalibrationDiagnostics.expected_calibration_error(
            y_true, y_proba, strategy="uniform"
        )
        assert 0.0 <= ece_q <= 1.0
        assert 0.0 <= ece_u <= 1.0

    def test_ece_invalid_strategy(self, binary_data):
        y_true, y_proba = binary_data
        with pytest.raises(ValueError, match="Unknown binning strategy"):
            CalibrationDiagnostics.expected_calibration_error(y_true, y_proba, strategy="invalid")

    def test_mce_strategies(self, binary_data):
        y_true, y_proba = binary_data
        for strategy in ("quantile", "uniform"):
            mce = CalibrationDiagnostics.maximum_calibration_error(
                y_true, y_proba, strategy=strategy
            )
            assert isinstance(mce, float)
            assert 0.0 <= mce <= 1.0

    def test_last_bin_includes_p_equals_one(self):
        """Samples with p == 1.0 must be counted in the last bin."""
        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([1.0, 1.0, 0.0, 0.0])
        for strategy in ("quantile", "uniform"):
            ece = CalibrationDiagnostics.expected_calibration_error(
                y_true, y_proba, n_bins=10, strategy=strategy
            )
            assert isinstance(ece, float)
            assert ece == pytest.approx(0.0, abs=1e-10)

    def test_reliability_diagram_strategies(self, binary_data):
        y_true, y_proba = binary_data
        for strategy in ("quantile", "uniform"):
            df = CalibrationDiagnostics.reliability_diagram_data(
                y_true, y_proba, strategy=strategy
            )
            assert isinstance(df, pd.DataFrame)
            assert "count" in df.columns
            assert df["count"].sum() == len(y_true)

    def test_reliability_diagram_counts_p_one(self):
        """p == 1.0 must appear in the bin counts."""
        y_true = np.array([1, 1, 0])
        y_proba = np.array([1.0, 0.5, 0.0])
        for strategy in ("quantile", "uniform"):
            df = CalibrationDiagnostics.reliability_diagram_data(
                y_true, y_proba, strategy=strategy
            )
            assert df["count"].sum() == 3

    def test_brier(self, binary_data):
        y_true, y_proba = binary_data
        brier = CalibrationDiagnostics.brier_score(y_true, y_proba)
        assert isinstance(brier, float)
        assert 0.0 <= brier <= 1.0

    def test_brier_decomposition(self, binary_data):
        y_true, y_proba = binary_data
        decomp = CalibrationDiagnostics.brier_decomposition(y_true, y_proba)
        assert isinstance(decomp, dict)
        assert "reliability" in decomp
        assert "resolution" in decomp
        assert "uncertainty" in decomp
        for v in decomp.values():
            assert isinstance(v, float)

    def test_brier_decomposition_strategies(self, binary_data):
        y_true, y_proba = binary_data
        for strategy in ("quantile", "uniform"):
            decomp = CalibrationDiagnostics.brier_decomposition(y_true, y_proba, strategy=strategy)
            assert "reliability" in decomp
            assert decomp["reliability"] >= 0

    def test_compare_before_after(self, binary_data):
        y_true, y_proba = binary_data
        cal_proba = np.column_stack([1 - y_proba, y_proba])
        raw_proba = np.column_stack([1 - y_proba, y_proba])
        result = CalibrationDiagnostics.compare_before_after(y_true, raw_proba, cal_proba)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert set(result["stage"].values) == {"raw", "calibrated"}
