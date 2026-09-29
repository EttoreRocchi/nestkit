"""Tests for survival target construction and normalization."""

import numpy as np
import pandas as pd
import pytest

from nestkit.survival._target import _normalize_survival_target, make_survival_target


class TestMakeSurvivalTarget:
    def test_valid_inputs(self):
        event = np.array([1, 0, 1, 1, 0])
        duration = np.array([5.0, 10.0, 3.0, 7.0, 12.0])
        y = make_survival_target(event, duration)
        assert y.shape == (5, 2)
        assert y.dtype == np.float64
        np.testing.assert_array_equal(y[:, 0], event)
        np.testing.assert_array_equal(y[:, 1], duration)

    def test_rejects_non_binary_event(self):
        event = np.array([0, 1, 2])
        duration = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="binary"):
            make_survival_target(event, duration)

    def test_rejects_negative_duration(self):
        event = np.array([1, 0, 1])
        duration = np.array([1.0, -2.0, 3.0])
        with pytest.raises(ValueError, match="positive"):
            make_survival_target(event, duration)

    def test_rejects_zero_duration(self):
        event = np.array([1, 0])
        duration = np.array([0.0, 3.0])
        with pytest.raises(ValueError, match="positive"):
            make_survival_target(event, duration)

    def test_rejects_mismatched_lengths(self):
        event = np.array([1, 0])
        duration = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="same length"):
            make_survival_target(event, duration)

    def test_converts_int_event_to_float(self):
        event = [1, 0, 1]
        duration = [5, 10, 3]
        y = make_survival_target(event, duration)
        assert y.dtype == np.float64


class TestNormalizeSurvivalTarget:
    def test_2column_ndarray(self):
        y_in = np.array([[1.0, 5.0], [0.0, 10.0]])
        y_out = _normalize_survival_target(y_in)
        assert y_out.shape == (2, 2)
        np.testing.assert_array_equal(y_in, y_out)

    def test_structured_array(self):
        dt = np.dtype([("event", np.float64), ("duration", np.float64)])
        y_in = np.array([(1.0, 5.0), (0.0, 10.0)], dtype=dt)
        y_out = _normalize_survival_target(y_in)
        assert y_out.shape == (2, 2)
        assert y_out[0, 0] == 1.0
        assert y_out[0, 1] == 5.0

    def test_dataframe(self):
        df = pd.DataFrame({"event": [1, 0, 1], "duration": [5.0, 10.0, 3.0]})
        y_out = _normalize_survival_target(df)
        assert y_out.shape == (3, 2)
        np.testing.assert_array_equal(y_out[:, 0], [1, 0, 1])

    def test_dataframe_missing_column(self):
        df = pd.DataFrame({"event": [1, 0], "time": [5.0, 10.0]})
        with pytest.raises(ValueError, match="duration"):
            _normalize_survival_target(df)

    def test_structured_array_missing_field(self):
        dt = np.dtype([("event", np.float64), ("time", np.float64)])
        y_in = np.array([(1.0, 5.0)], dtype=dt)
        with pytest.raises(ValueError, match="duration"):
            _normalize_survival_target(y_in)

    def test_1d_array_rejected(self):
        with pytest.raises(ValueError, match="Survival target must be"):
            _normalize_survival_target(np.array([1.0, 2.0, 3.0]))

    def test_3column_array_rejected(self):
        with pytest.raises(ValueError, match="Survival target must be"):
            _normalize_survival_target(np.ones((5, 3)))
