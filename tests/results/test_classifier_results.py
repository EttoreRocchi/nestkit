"""Tests for ClassifierResults."""

import numpy as np

from nestkit.results.classifier_results import ClassifierOuterFoldResult, ClassifierResults


def _make_fold_result(fold_idx, n_samples=30, rng_seed=None):
    """Create a minimal mock ClassifierOuterFoldResult."""
    rng = np.random.RandomState(rng_seed if rng_seed is not None else fold_idx)
    y_true = rng.randint(0, 2, size=n_samples)
    y_proba_raw = np.column_stack([1 - y_true * 0.8, y_true * 0.8])
    y_pred_default = (y_proba_raw[:, 1] >= 0.5).astype(int)
    test_indices = np.arange(fold_idx * n_samples, (fold_idx + 1) * n_samples)
    train_indices = np.arange(200)  # dummy

    return ClassifierOuterFoldResult(
        fold_idx=fold_idx,
        train_indices=train_indices,
        test_indices=test_indices,
        best_params={"C": 1.0, "kernel": "rbf"},
        best_inner_score=0.9,
        inner_cv_results={
            "mean_test_score": [0.85, 0.90, 0.88],
            "std_test_score": [0.02, 0.01, 0.03],
            "rank_test_score": [3, 1, 2],
            "param_C": [0.1, 1.0, 10.0],
            "param_kernel": ["rbf", "rbf", "linear"],
        },
        fit_time=0.1,
        score_time=0.01,
        fitted_estimator=None,
        y_true=y_true,
        y_proba_raw=y_proba_raw,
        y_pred_default=y_pred_default,
        outer_scores_default={"accuracy": 0.85 + fold_idx * 0.02, "f1": 0.80 + fold_idx * 0.02},
        confusion_matrix_default=np.array([[10, 2], [3, 15]]),
    )


class TestClassifierResults:
    def test_finalize(self):
        results = ClassifierResults(n_outer_folds=3)
        for i in range(3):
            results.add_fold(_make_fold_result(i))
        results.finalize()
        assert hasattr(results, "summary_default_")
        assert len(results.summary_default_) > 0

    def test_has_calibration(self):
        results = ClassifierResults(n_outer_folds=2)
        for i in range(2):
            results.add_fold(_make_fold_result(i))
        assert results.has_calibration is False

    def test_has_threshold(self):
        results = ClassifierResults(n_outer_folds=2)
        for i in range(2):
            results.add_fold(_make_fold_result(i))
        assert results.has_threshold_optimization is False

    def test_predictions_dataframe(self):
        results = ClassifierResults(n_outer_folds=3)
        for i in range(3):
            results.add_fold(_make_fold_result(i))
        results.finalize()
        preds = results.predictions_
        assert "y_true" in preds.columns
        assert "y_pred_default" in preds.columns
        assert "fold_idx" in preds.columns
