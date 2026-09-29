"""Binary targets must behave identically whatever the class labels are called.

The binary path used to build predictions with ``.astype(int)`` instead of
indexing ``classes_``, so anything other than a literal ``{0, 1}`` target
crashed inside scikit-learn's metrics.  Behind that crash sat three more
label-dependent defects: isotonic calibration mapped every sample to 1.0,
Youden's J returned -1 at every threshold, and the ECE left its ``[0, 1]``
range.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from nestkit import NestedCVClassifier


@pytest.fixture
def binary_xy():
    rng = np.random.RandomState(0)
    X = rng.randn(240, 4)
    y = (X[:, 0] + rng.randn(240) * 0.7 > 0).astype(int)
    return X, y


def _fit(X, y, **kwargs):
    return NestedCVClassifier(
        estimator=LogisticRegression(max_iter=500),
        param_grid={"C": [0.1, 1.0]},
        outer_cv=4,
        inner_cv=2,
        random_state=0,
        **kwargs,
    ).fit(X, y)


ORIENTATION_PRESERVING = [
    pytest.param(lambda y: y, id="0-1"),
    pytest.param(lambda y: y + 1, id="1-2"),
    pytest.param(lambda y: y * 2 + 3, id="3-5"),
    pytest.param(lambda y: np.where(y == 1, "pos", "neg"), id="neg-pos"),
    pytest.param(lambda y: np.where(y == 1, "yes", "no"), id="no-yes"),
]


class TestLabelInvariance:
    @pytest.mark.parametrize("relabel", ORIENTATION_PRESERVING)
    def test_metrics_are_unchanged(self, binary_xy, relabel):
        X, y = binary_xy
        baseline = _fit(X, y).results_.summary_default_.set_index("metric")["mean"]
        renamed = _fit(X, relabel(y)).results_.summary_default_.set_index("metric")["mean"]
        assert np.allclose(baseline.to_numpy(), renamed.to_numpy(), equal_nan=True)

    @pytest.mark.parametrize("relabel", ORIENTATION_PRESERVING)
    def test_calibration_and_thresholds_are_unchanged(self, binary_xy, relabel):
        """Covers the calibrator fit, the threshold search and the ECE."""
        X, y = binary_xy
        kwargs = dict(
            calibration_method="isotonic",
            threshold_strategy="pooled",
            threshold_criterion="youden",
        )
        baseline = _fit(X, y, **kwargs).results_
        renamed = _fit(X, relabel(y), **kwargs).results_

        assert np.allclose(baseline.thresholds_per_fold_, renamed.thresholds_per_fold_)

        def eces(results):
            return [
                fr.oof_calibration_diagnostics["ece_calibrated"]
                for fr in results.fold_results_
                if fr.oof_calibration_diagnostics
            ]

        assert np.allclose(eces(baseline), eces(renamed))

    @pytest.mark.parametrize("relabel", ORIENTATION_PRESERVING)
    def test_conformal_coverage_is_unchanged(self, binary_xy, relabel):
        X, y = binary_xy
        baseline = _fit(X, y, conformal_prediction=True).results_
        renamed = _fit(X, relabel(y), conformal_prediction=True).results_
        assert baseline.conformal_coverage_["mean"] == pytest.approx(
            renamed.conformal_coverage_["mean"]
        )


class TestOriginalLabelsAreReturned:
    @pytest.mark.parametrize("relabel", ORIENTATION_PRESERVING)
    def test_classes_and_predictions(self, binary_xy, relabel):
        X, y = binary_xy
        y_relabelled = relabel(y)
        ncv = _fit(X, y_relabelled, threshold_strategy="pooled")

        assert set(ncv.classes_) == set(np.unique(y_relabelled))
        for fold in ncv.results_.fold_results_:
            assert set(np.unique(fold.y_pred_default)) <= set(ncv.classes_)
            assert set(np.unique(fold.y_true)) <= set(ncv.classes_)
            assert set(np.unique(fold.y_pred_optimized)) <= set(ncv.classes_)

    def test_predictions_dataframe_uses_original_labels(self, binary_xy):
        X, y = binary_xy
        ncv = _fit(X, np.where(y == 1, "pos", "neg"))
        preds = ncv.results_.predictions_
        assert set(preds["y_true"].unique()) <= {"neg", "pos"}
        assert set(preds["y_pred_default"].unique()) <= {"neg", "pos"}

    def test_conformal_sets_hold_original_labels(self, binary_xy):
        X, y = binary_xy
        ncv = _fit(X, np.where(y == 1, "pos", "neg"), conformal_prediction=True)
        for fold in ncv.results_.fold_results_:
            for prediction_set in fold.conformal_prediction_sets:
                assert set(prediction_set) <= {"neg", "pos"}


class TestPositiveClassConvention:
    def test_second_sorted_class_is_positive(self, binary_xy):
        """``classes_[1]`` is the positive class, as in scikit-learn.

        ``{'case', 'control'}`` therefore treats ``'control'`` as positive,
        which flips asymmetric metrics such as recall.  This is a convention,
        not a defect, and the symmetric ROC AUC stays put.
        """
        X, y = binary_xy
        baseline = _fit(X, y).results_.summary_default_.set_index("metric")["mean"]
        flipped = _fit(X, np.where(y == 1, "case", "control")).results_.summary_default_.set_index(
            "metric"
        )["mean"]
        assert flipped["roc_auc"] == pytest.approx(baseline["roc_auc"])
        assert flipped["recall"] != pytest.approx(baseline["recall"])
