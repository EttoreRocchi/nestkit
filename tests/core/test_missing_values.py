"""Missing values in X: imputation must be fitted inside the folds."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline

from nestkit.classifier import NestedCVClassifier
from nestkit.regressor import NestedCVRegressor

pytestmark = pytest.mark.slow


def _add_nan(X, rate=0.1, seed=0):
    """Return a copy of X with a fraction of entries set to NaN."""
    X = X.astype(float).copy()
    X[np.random.default_rng(seed).random(X.shape) < rate] = np.nan
    return X


@pytest.fixture
def binary_data_nan():
    X, y = make_classification(n_samples=150, n_features=6, random_state=0)
    return _add_nan(X), y


def _imputing_classifier(**kwargs):
    return NestedCVClassifier(
        estimator=make_pipeline(SimpleImputer(), LogisticRegression()),
        param_grid={"logisticregression__C": [0.1, 1.0]},
        outer_cv=3,
        inner_cv=3,
        **kwargs,
    )


class TestNaNAccepted:
    def test_classifier_with_imputer_pipeline(self, binary_data_nan):
        """Verify NaN in X is accepted when the pipeline imputes it."""
        X, y = binary_data_nan
        ncv = _imputing_classifier(
            calibration_method="sigmoid",
            threshold_strategy="fold_specific",
            conformal_prediction=True,
        ).fit(X, y)
        assert len(ncv.results_.fold_results_) == 3

    def test_regressor_with_imputer_pipeline(self):
        """Verify NaN in X is accepted by NestedCVRegressor."""
        X, y = make_regression(n_samples=150, n_features=5, random_state=0)
        ncv = NestedCVRegressor(
            estimator=make_pipeline(SimpleImputer(), Ridge()),
            param_grid={"ridge__alpha": [0.1, 1.0]},
            outer_cv=3,
            inner_cv=3,
            mondrian_bins=2,
        ).fit(_add_nan(X), y)
        assert len(ncv.results_.fold_results_) == 3

    def test_pandas_nullable_dtype(self, binary_data_nan):
        """Verify pd.NA in nullable dtypes is treated as NaN and names are kept."""
        X, y = binary_data_nan
        X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])]).astype("Float64")
        assert X_df.isna().to_numpy().any()
        ncv = _imputing_classifier().fit(X_df, y)
        assert ncv.feature_names_in_ == list(X_df.columns)

    def test_estimator_without_nan_support_raises(self, binary_data_nan):
        """Verify an estimator that cannot handle NaN still fails loudly."""
        X, y = binary_data_nan
        ncv = NestedCVClassifier(
            estimator=LogisticRegression(),
            param_grid={"C": [1.0]},
            outer_cv=3,
            inner_cv=3,
        )
        with pytest.raises(ValueError, match="NaN"):
            ncv.fit(X, y)


class TestStillRejected:
    def test_inf_in_X(self, binary_data_nan):
        """Verify infinite values in X are rejected."""
        X, y = binary_data_nan
        X[0, 0] = np.inf
        with pytest.raises(ValueError, match="infinity"):
            _imputing_classifier().fit(X, y)

    def test_nan_in_y(self):
        """Verify NaN in y is rejected."""
        X, y = make_regression(n_samples=60, n_features=3, random_state=0)
        y[0] = np.nan
        ncv = NestedCVRegressor(
            estimator=make_pipeline(SimpleImputer(), Ridge()),
            param_grid={"ridge__alpha": [1.0]},
            outer_cv=3,
            inner_cv=3,
        )
        with pytest.raises(ValueError, match="Input y contains NaN"):
            ncv.fit(_add_nan(X), y)


class TestImputationLeakage:
    def test_outer_imputer_statistics_from_train_rows_only(self, binary_data_nan):
        """Verify each outer-fold imputer is fitted on the outer training rows only."""
        X, y = binary_data_nan
        ncv = _imputing_classifier().fit(X, y)

        for fr in ncv.results_.fold_results_:
            imputer = fr.fitted_estimator[0]
            np.testing.assert_allclose(
                imputer.statistics_, np.nanmean(X[fr.train_indices], axis=0)
            )
            assert not np.allclose(imputer.statistics_, np.nanmean(X, axis=0))

    def test_no_imputer_fit_sees_outer_test_rows(self, binary_data_nan):
        """Verify no imputer fit (inner search, calibration, refit) sees outer-test rows."""
        X, y = binary_data_nan
        # Column 0 is a NaN-free row identifier
        X[:, 0] = np.arange(len(X))

        outer_test = {}
        current_fold = []
        fits = []

        class _FoldTracker:
            def on_outer_fold_start(self, fold_idx, train_idx, test_idx):
                outer_test[fold_idx] = set(test_idx)
                current_fold.append(fold_idx)

        original_fit = SimpleImputer.fit

        def _spy_fit(self, X_fit, y_fit=None):
            fits.append((current_fold[-1], set(X_fit[:, 0].astype(int))))
            return original_fit(self, X_fit, y_fit)

        ncv = _imputing_classifier(calibration_method="isotonic", callbacks=[_FoldTracker()])
        with patch.object(SimpleImputer, "fit", _spy_fit):
            ncv.fit(X, y)

        assert {fold for fold, _ in fits} == {0, 1, 2}
        for fold_idx, seen_rows in fits:
            assert not seen_rows & outer_test[fold_idx], (
                f"Fold {fold_idx}: imputer fitted on outer-test rows"
            )
