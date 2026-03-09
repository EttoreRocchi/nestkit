"""Extended tests for calibration diagnostics and calibrators.

Covers additional edge cases, boundary values, and error paths not in
tests/calibration/test_calibrators.py.
"""

import numpy as np
import pandas as pd
import pytest

from nestkit.calibration.calibrators import PostHocCalibrator
from nestkit.calibration.diagnostics import CalibrationDiagnostics


@pytest.fixture
def well_calibrated_data():
    """Data where predictions closely match true frequencies."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=200)
    y_proba = np.clip(y_true + rng.normal(0, 0.2, size=200), 0.01, 0.99)
    return y_true, y_proba


class TestCalibrationDiagnosticsExtended:
    def test_reliability_diagram_columns(self, well_calibrated_data):
        y_true, y_proba = well_calibrated_data
        df = CalibrationDiagnostics.reliability_diagram_data(y_true, y_proba, n_bins=5)
        expected_cols = {
            "bin_lower",
            "bin_upper",
            "bin_mid",
            "mean_predicted",
            "fraction_positive",
            "count",
        }
        assert set(df.columns) == expected_cols

    def test_reliability_diagram_bin_count_uniform(self, well_calibrated_data):
        y_true, y_proba = well_calibrated_data
        df = CalibrationDiagnostics.reliability_diagram_data(
            y_true, y_proba, n_bins=5, strategy="uniform"
        )
        assert len(df) == 5

    def test_reliability_diagram_empty_bins_uniform(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.9, 0.9])
        df = CalibrationDiagnostics.reliability_diagram_data(
            y_true, y_proba, n_bins=10, strategy="uniform"
        )
        empty = df[df["count"] == 0]
        assert len(empty) > 0
        assert empty["mean_predicted"].isna().all()
        assert empty["fraction_positive"].isna().all()

    def test_reliability_diagram_quantile_all_nonempty(self, well_calibrated_data):
        y_true, y_proba = well_calibrated_data
        df = CalibrationDiagnostics.reliability_diagram_data(y_true, y_proba, n_bins=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_reliability_diagram_single_sample(self):
        y_true = np.array([1])
        y_proba = np.array([0.8])
        df = CalibrationDiagnostics.reliability_diagram_data(
            y_true, y_proba, n_bins=5, strategy="uniform"
        )
        assert isinstance(df, pd.DataFrame)
        assert df["count"].sum() == 1

    def test_mce_perfect_calibration(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        mce = CalibrationDiagnostics.maximum_calibration_error(y_true, y_proba, n_bins=2)
        assert mce == pytest.approx(0.0, abs=0.01)

    def test_mce_worst_case(self):
        y_true = np.zeros(10)
        y_proba = np.full(10, 0.95)
        mce = CalibrationDiagnostics.maximum_calibration_error(
            y_true, y_proba, n_bins=10, strategy="uniform"
        )
        assert mce > 0.9

    def test_mce_geq_ece(self, well_calibrated_data):
        y_true, y_proba = well_calibrated_data
        ece = CalibrationDiagnostics.expected_calibration_error(y_true, y_proba)
        mce = CalibrationDiagnostics.maximum_calibration_error(y_true, y_proba)
        assert mce >= ece - 1e-10

    def test_ece_2d_proba_input(self, well_calibrated_data):
        y_true, y_proba = well_calibrated_data
        y_proba_2d = np.column_stack([1 - y_proba, y_proba])
        ece1 = CalibrationDiagnostics.expected_calibration_error(y_true, y_proba)
        ece2 = CalibrationDiagnostics.expected_calibration_error(y_true, y_proba_2d)
        assert ece1 == pytest.approx(ece2)

    def test_brier_decomposition_relation(self, well_calibrated_data):
        y_true, y_proba = well_calibrated_data
        brier = CalibrationDiagnostics.brier_score(y_true, y_proba)
        decomp = CalibrationDiagnostics.brier_decomposition(y_true, y_proba)
        reconstructed = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"]
        assert brier == pytest.approx(reconstructed, abs=0.05)

    def test_brier_score_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.0, 0.0, 1.0, 1.0])
        assert CalibrationDiagnostics.brier_score(y_true, y_proba) == 0.0

    def test_brier_score_worst_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([1.0, 1.0, 0.0, 0.0])
        assert CalibrationDiagnostics.brier_score(y_true, y_proba) == 1.0

    def test_compare_before_after_columns(self, well_calibrated_data):
        y_true, y_proba = well_calibrated_data
        df = CalibrationDiagnostics.compare_before_after(y_true, y_proba, y_proba)
        assert set(df.columns) == {"stage", "ece", "mce", "brier"}
        assert list(df["stage"]) == ["raw", "calibrated"]


class TestBrierDecompositionIdentity:
    """Verify BS = Reliability - Resolution + Uncertainty across configurations."""

    @pytest.mark.parametrize("n_bins", [5, 10, 20, 50])
    def test_identity_uniform_bins(self, n_bins):
        rng = np.random.RandomState(123)
        y_true = rng.randint(0, 2, size=500)
        y_proba = np.clip(y_true + rng.normal(0, 0.3, size=500), 0.01, 0.99)
        bs = CalibrationDiagnostics.brier_score(y_true, y_proba)
        decomp = CalibrationDiagnostics.brier_decomposition(
            y_true, y_proba, n_bins=n_bins, strategy="uniform"
        )
        reconstructed = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"]
        assert bs == pytest.approx(reconstructed, abs=0.02)

    @pytest.mark.parametrize("n_bins", [10, 20, 50])
    def test_identity_quantile_bins(self, n_bins):
        rng = np.random.RandomState(456)
        y_true = rng.randint(0, 2, size=500)
        y_proba = np.clip(y_true + rng.normal(0, 0.25, size=500), 0.01, 0.99)
        bs = CalibrationDiagnostics.brier_score(y_true, y_proba)
        decomp = CalibrationDiagnostics.brier_decomposition(
            y_true, y_proba, n_bins=n_bins, strategy="quantile"
        )
        reconstructed = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"]
        assert bs == pytest.approx(reconstructed, abs=0.02)

    def test_identity_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.0, 0.0, 1.0, 1.0])
        bs = CalibrationDiagnostics.brier_score(y_true, y_proba)
        decomp = CalibrationDiagnostics.brier_decomposition(y_true, y_proba, n_bins=2)
        reconstructed = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"]
        assert bs == pytest.approx(0.0)
        assert reconstructed == pytest.approx(0.0, abs=0.01)


class TestPostHocCalibratorExtended:
    def test_predict_before_fit_raises_runtime_error(self):
        cal = PostHocCalibrator("sigmoid")
        with pytest.raises(RuntimeError, match="not fitted"):
            cal.predict_proba(np.array([0.5, 0.6]))

    def test_unknown_method_raises_valueerror_on_fit(self):
        cal = PostHocCalibrator("platt")
        with pytest.raises(ValueError, match="Unknown calibration method"):
            cal.fit(np.array([0.5, 0.6]), np.array([0, 1]))

    @pytest.mark.parametrize("method", ["sigmoid", "isotonic", "beta"])
    def test_calibrated_probabilities_in_01(self, method, well_calibrated_data):
        y_true, y_proba = well_calibrated_data
        cal = PostHocCalibrator(method)
        cal.fit(y_proba, y_true)
        result = cal.predict_proba(y_proba)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    @pytest.mark.parametrize("method", ["sigmoid", "isotonic", "beta"])
    def test_calibrated_probabilities_sum_to_one(self, method, well_calibrated_data):
        y_true, y_proba = well_calibrated_data
        cal = PostHocCalibrator(method)
        cal.fit(y_proba, y_true)
        result = cal.predict_proba(y_proba)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)

    def test_venn_abers_small_dataset(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_proba = np.array([0.2, 0.8, 0.3, 0.7, 0.9])
        cal = PostHocCalibrator("venn_abers")
        cal.fit(y_proba, y_true)
        result = cal.predict_proba(y_proba[:3])
        assert result.shape == (3, 2)

    def test_2d_proba_input_handled(self, well_calibrated_data):
        y_true, y_proba = well_calibrated_data
        y_proba_2d = np.column_stack([1 - y_proba, y_proba])
        cal = PostHocCalibrator("sigmoid")
        cal.fit(y_proba_2d, y_true)
        result = cal.predict_proba(y_proba_2d)
        assert result.shape == (len(y_proba), 2)

    def test_fit_returns_self(self, well_calibrated_data):
        y_true, y_proba = well_calibrated_data
        cal = PostHocCalibrator("sigmoid")
        returned = cal.fit(y_proba, y_true)
        assert returned is cal
