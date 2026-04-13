"""Tests for MondrianClassifierConformal -- CV+ Mondrian conformal prediction."""

import numpy as np
import pytest

from nestkit.conformal.classifier_conformal import MondrianClassifierConformal
from nestkit.conformal.results import ClassifierConformalResult


@pytest.fixture()
def binary_data():
    """Well-separated binary classification data (n=200)."""
    rng = np.random.RandomState(42)
    n = 200
    y = np.array([0] * 100 + [1] * 100)
    probas = np.zeros((n, 2))
    # Good predictions: high probability for the true class
    probas[y == 0, 0] = rng.uniform(0.7, 0.95, size=100)
    probas[y == 0, 1] = 1 - probas[y == 0, 0]
    probas[y == 1, 1] = rng.uniform(0.7, 0.95, size=100)
    probas[y == 1, 0] = 1 - probas[y == 1, 1]
    return probas, y


@pytest.fixture()
def multiclass_data():
    """3-class data (n=300)."""
    rng = np.random.RandomState(42)
    n_per_class = 100
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)
    probas = rng.dirichlet([5, 1, 1], size=n_per_class)  # class 0 dominant
    probas = np.vstack(
        [
            probas,
            rng.dirichlet([1, 5, 1], size=n_per_class),  # class 1 dominant
            rng.dirichlet([1, 1, 5], size=n_per_class),  # class 2 dominant
        ]
    )
    return probas, y


class TestFit:
    def test_binary_returns_correct_shape(self, binary_data):
        probas, y = binary_data
        result = MondrianClassifierConformal.fit(probas, y, np.array([0, 1]))
        assert isinstance(result, ClassifierConformalResult)
        assert result.qhat_per_class.shape == (2,)
        assert result.n_calibration_per_class.shape == (2,)

    def test_binary_qhat_in_unit_interval(self, binary_data):
        probas, y = binary_data
        result = MondrianClassifierConformal.fit(probas, y, np.array([0, 1]))
        assert np.all(result.qhat_per_class >= 0)
        assert np.all(result.qhat_per_class <= 1)

    def test_multiclass_returns_correct_shape(self, multiclass_data):
        probas, y = multiclass_data
        result = MondrianClassifierConformal.fit(probas, y, np.array([0, 1, 2]))
        assert result.qhat_per_class.shape == (3,)
        assert result.n_calibration_per_class.shape == (3,)
        np.testing.assert_array_equal(result.n_calibration_per_class, [100, 100, 100])

    def test_alpha_stored(self, binary_data):
        probas, y = binary_data
        result = MondrianClassifierConformal.fit(probas, y, np.array([0, 1]), alpha=0.2)
        assert result.alpha == 0.2

    def test_good_predictions_give_low_qhat(self, binary_data):
        probas, y = binary_data
        result = MondrianClassifierConformal.fit(probas, y, np.array([0, 1]))
        # With well-separated data, q_hat should be relatively low
        assert np.all(result.qhat_per_class < 0.5)

    def test_empty_class_defaults_to_one(self):
        """Class with 0 samples gets q_hat=1.0 (conservative)."""
        probas = np.array([[0.9, 0.1], [0.85, 0.15], [0.8, 0.2]])
        y = np.array([0, 0, 0])  # No class 1
        with pytest.warns(UserWarning, match="no calibration samples"):
            result = MondrianClassifierConformal.fit(probas, y, np.array([0, 1]))
        assert result.qhat_per_class[1] == 1.0
        assert result.n_calibration_per_class[1] == 0

    def test_single_sample_class(self):
        """Class with 1 sample gets q_hat=1.0."""
        probas = np.array([[0.9, 0.1], [0.85, 0.15], [0.2, 0.8]])
        y = np.array([0, 0, 1])
        result = MondrianClassifierConformal.fit(probas, y, np.array([0, 1]))
        assert result.qhat_per_class[1] == 1.0


class TestPredict:
    def test_binary_set_sizes(self, binary_data):
        probas, y = binary_data
        result = MondrianClassifierConformal.fit(probas, y, np.array([0, 1]))
        output = MondrianClassifierConformal.predict(probas, result)

        assert output["set_sizes"].shape == (len(y),)
        assert set(np.unique(output["set_sizes"])).issubset({0, 1, 2})
        assert len(output["prediction_sets"]) == len(y)

    def test_prediction_set_contents_match_sizes(self, binary_data):
        probas, y = binary_data
        result = MondrianClassifierConformal.fit(probas, y, np.array([0, 1]))
        output = MondrianClassifierConformal.predict(probas, result)

        for i in range(len(y)):
            assert len(output["prediction_sets"][i]) == output["set_sizes"][i]

    def test_is_uncertain_matches_sizes(self, binary_data):
        probas, y = binary_data
        result = MondrianClassifierConformal.fit(probas, y, np.array([0, 1]))
        output = MondrianClassifierConformal.predict(probas, result)

        np.testing.assert_array_equal(output["is_uncertain"], output["set_sizes"] > 1)
        np.testing.assert_array_equal(output["is_empty"], output["set_sizes"] == 0)

    def test_multiclass_set_sizes_range(self, multiclass_data):
        probas, y = multiclass_data
        result = MondrianClassifierConformal.fit(probas, y, np.array([0, 1, 2]))
        output = MondrianClassifierConformal.predict(probas, result)

        assert np.all(output["set_sizes"] >= 0)
        assert np.all(output["set_sizes"] <= 3)


class TestCoverageGuarantee:
    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
    def test_binary_coverage(self, alpha):
        """With enough data, empirical coverage should be near 1-alpha."""
        rng = np.random.RandomState(0)
        n = 500
        y = rng.randint(0, 2, size=n)
        probas = np.zeros((n, 2))
        probas[y == 0, 0] = rng.uniform(0.6, 0.95, size=(y == 0).sum())
        probas[y == 0, 1] = 1 - probas[y == 0, 0]
        probas[y == 1, 1] = rng.uniform(0.6, 0.95, size=(y == 1).sum())
        probas[y == 1, 0] = 1 - probas[y == 1, 1]

        result = MondrianClassifierConformal.fit(probas, y, np.array([0, 1]), alpha=alpha)
        output = MondrianClassifierConformal.predict(probas, result)

        coverage = np.mean([int(y[i]) in output["prediction_sets"][i] for i in range(n)])
        assert coverage >= 1 - alpha - 0.05  # small tolerance

    def test_multiclass_coverage(self, multiclass_data):
        probas, y = multiclass_data
        alpha = 0.1
        result = MondrianClassifierConformal.fit(probas, y, np.array([0, 1, 2]), alpha=alpha)
        output = MondrianClassifierConformal.predict(probas, result)

        coverage = np.mean([int(y[i]) in output["prediction_sets"][i] for i in range(len(y))])
        assert coverage >= 1 - alpha - 0.05


class TestNonZeroIndexedLabels:
    """Regression tests for labels that don't match their column indices."""

    def test_predict_with_classes_returns_labels(self):
        """When classes is passed, prediction sets should contain class labels, not indices."""
        rng = np.random.RandomState(42)
        n = 200
        classes = np.array([5, 10])
        y = rng.choice(classes, size=n)
        probas = np.zeros((n, 2))
        probas[y == 5, 0] = rng.uniform(0.7, 0.95, size=(y == 5).sum())
        probas[y == 5, 1] = 1 - probas[y == 5, 0]
        probas[y == 10, 1] = rng.uniform(0.7, 0.95, size=(y == 10).sum())
        probas[y == 10, 0] = 1 - probas[y == 10, 1]

        result = MondrianClassifierConformal.fit(probas, y, classes, alpha=0.1)
        output = MondrianClassifierConformal.predict(probas, result, classes=classes)

        # Prediction sets should contain actual class labels (5, 10), not indices (0, 1)
        all_labels_in_sets = set()
        for ps in output["prediction_sets"]:
            all_labels_in_sets.update(ps)
        assert all_labels_in_sets.issubset({5, 10})
        assert 0 not in all_labels_in_sets

    def test_coverage_with_shifted_labels(self):
        """Coverage should be correct even when labels are {1, 2, 3}."""
        rng = np.random.RandomState(0)
        n = 500
        classes = np.array([1, 2, 3])
        y = rng.choice(classes, size=n)
        probas = np.zeros((n, 3))
        for c_idx, c in enumerate(classes):
            mask = y == c
            probas[mask, c_idx] = rng.uniform(0.6, 0.9, size=mask.sum())
            # Distribute remaining probability
            remaining = 1 - probas[mask, c_idx]
            for other in range(3):
                if other != c_idx:
                    probas[mask, other] = remaining / 2

        alpha = 0.1
        result = MondrianClassifierConformal.fit(probas, y, classes, alpha=alpha)
        output = MondrianClassifierConformal.predict(probas, result, classes=classes)

        coverage = np.mean([y[i] in output["prediction_sets"][i] for i in range(n)])
        assert coverage >= 1 - alpha - 0.05

    def test_predict_without_classes_returns_indices(self):
        """Without classes parameter, prediction sets should still contain indices."""
        probas = np.array([[0.9, 0.1], [0.2, 0.8]])
        y = np.array([0, 1])
        result = MondrianClassifierConformal.fit(probas, y, np.array([0, 1]), alpha=0.1)
        output = MondrianClassifierConformal.predict(probas, result)

        all_labels_in_sets = set()
        for ps in output["prediction_sets"]:
            all_labels_in_sets.update(ps)
        assert all_labels_in_sets.issubset({0, 1})
