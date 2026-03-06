"""Tests for nestkit._validation -- input validation helpers.

Covers: validate_threshold_params, validate_calibration_method,
        ensure_2d_proba, extract_positive_proba.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nestkit._validation import (
    ensure_2d_proba,
    extract_positive_proba,
    validate_calibration_method,
    validate_threshold_params,
)


class TestValidateThresholdParams:
    @pytest.mark.parametrize("strategy", [None, "fold_specific", "pooled"])
    def test_valid_strategy_accepted(self, strategy):
        validate_threshold_params(strategy, "youden", None, None)

    @pytest.mark.parametrize(
        "criterion,cost_matrix,min_recall",
        [
            ("youden", None, None),
            ("f_beta", None, None),
            ("cost", [[0, 1], [5, 0]], None),
            ("balanced_accuracy", None, None),
            ("precision_at_recall", None, 0.9),
        ],
    )
    def test_valid_string_criterion_accepted(self, criterion, cost_matrix, min_recall):
        validate_threshold_params("pooled", criterion, cost_matrix, min_recall)

    def test_callable_criterion_accepted(self):
        validate_threshold_params("pooled", lambda y, p, t: 0.5, None, None)

    def test_invalid_strategy_raises_valueerror(self):
        with pytest.raises(ValueError, match="threshold_strategy"):
            validate_threshold_params("invalid", "youden", None, None)

    def test_invalid_string_criterion_raises_valueerror(self):
        with pytest.raises(ValueError, match="threshold_criterion"):
            validate_threshold_params("pooled", "unknown", None, None)

    def test_cost_criterion_without_matrix_raises_valueerror(self):
        with pytest.raises(ValueError, match="cost_matrix is required"):
            validate_threshold_params("pooled", "cost", None, None)

    def test_precision_at_recall_without_min_recall_raises_valueerror(self):
        with pytest.raises(ValueError, match="min_recall is required"):
            validate_threshold_params("pooled", "precision_at_recall", None, None)

    def test_cost_criterion_with_matrix_passes(self):
        validate_threshold_params("pooled", "cost", np.array([[0, 1], [5, 0]]), None)

    def test_precision_at_recall_with_min_recall_passes(self):
        validate_threshold_params("pooled", "precision_at_recall", None, 0.9)


class TestValidateCalibrationMethod:
    @pytest.mark.parametrize("method", [None, "sigmoid", "isotonic", "beta", "venn_abers"])
    def test_valid_method_accepted(self, method):
        validate_calibration_method(method)

    def test_invalid_method_raises_valueerror(self):
        with pytest.raises(ValueError, match="calibration_method"):
            validate_calibration_method("platt")


class TestEnsure2dProba:
    def test_1d_input_becomes_2d(self):
        p = np.array([0.1, 0.5, 0.9])
        result = ensure_2d_proba(p)
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result.sum(axis=1), 1.0)

    def test_2d_input_unchanged(self):
        p = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
        result = ensure_2d_proba(p)
        np.testing.assert_array_equal(result, p)

    def test_single_element_1d(self):
        p = np.array([0.7])
        result = ensure_2d_proba(p)
        assert result.shape == (1, 2)

    @given(arrays(float, shape=st.integers(1, 50), elements=st.floats(0, 1)))
    def test_ensure_2d_always_returns_2d(self, arr):
        result = ensure_2d_proba(arr)
        assert result.ndim == 2
        assert result.shape[0] == len(arr)

    @given(arrays(float, shape=st.integers(1, 50), elements=st.floats(0, 1)))
    def test_columns_sum_to_one(self, arr):
        result = ensure_2d_proba(arr)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-14)


class TestExtractPositiveProba:
    def test_2d_returns_second_column(self):
        p = np.array([[0.3, 0.7], [0.6, 0.4]])
        result = extract_positive_proba(p)
        np.testing.assert_array_equal(result, np.array([0.7, 0.4]))

    def test_1d_returned_unchanged(self):
        p = np.array([0.1, 0.5, 0.9])
        result = extract_positive_proba(p)
        np.testing.assert_array_equal(result, p)

    def test_single_element_2d(self):
        p = np.array([[0.4, 0.6]])
        result = extract_positive_proba(p)
        assert result.shape == (1,)
        assert result[0] == 0.6

    @given(
        arrays(
            float,
            shape=st.tuples(st.integers(1, 50), st.just(2)),
            elements=st.floats(0, 1),
        )
    )
    def test_extract_positive_from_2d_equals_column_1(self, arr):
        result = extract_positive_proba(arr)
        np.testing.assert_array_equal(result, arr[:, 1])
