"""Scikit-learn compatible wrapper for lifelines' CoxPHFitter.

Adapts the lifelines CoxPH API to the standard sklearn
``fit(X, y)`` / ``predict(X)`` interface so that the wrapper can
be used seamlessly with :class:`~sklearn.model_selection.GridSearchCV`,
:class:`~sklearn.model_selection.RandomizedSearchCV`, and
:class:`~nestkit.NestedCVSurvival`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def _check_lifelines():
    """Lazy-import lifelines with a helpful error message."""
    try:
        import lifelines

        return lifelines
    except ImportError:
        raise ImportError(
            "Survival analysis requires lifelines. Install it with: pip install nestkit[survival]"
        ) from None


class CoxPHWrapper(BaseEstimator):
    """Scikit-learn compatible wrapper around :class:`lifelines.CoxPHFitter`.

    Wraps the lifelines Cox proportional hazards model to follow the
    standard sklearn ``fit`` / ``predict`` / ``score`` interface.
    The target ``y`` is expected to be a 2-column array where
    ``y[:, 0]`` is the binary event indicator and ``y[:, 1]`` is the
    observed duration.

    Parameters
    ----------
    penalizer : float, default=0.0
        L1/L2 penalty strength for regularization.
    l1_ratio : float, default=0.0
        Elastic-net mixing parameter (0 = pure L2, 1 = pure L1).
    alpha : float, default=0.05
        Significance level for confidence intervals.  Not a tunable
        hyperparameter; it affects only inference output.
    baseline_estimation_method : str, default='breslow'
        Method for baseline hazard estimation (``'breslow'`` or
        ``'spline'``).  Not a tunable hyperparameter; it affects
        baseline hazard estimation.

    Attributes
    ----------
    fitter_ : lifelines.CoxPHFitter
        Fitted CoxPHFitter instance (available after ``fit``).
    feature_names_in_ : list of str
        Feature names seen during ``fit``.
    n_features_in_ : int
        Number of features seen during ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from nestkit.survival import CoxPHWrapper, make_survival_target
    >>> X = np.random.randn(100, 5)
    >>> event = np.random.binomial(1, 0.7, 100)
    >>> duration = np.random.exponential(10, 100)
    >>> y = make_survival_target(event, duration)
    >>> model = CoxPHWrapper(penalizer=0.1)
    >>> model.fit(X, y)  # doctest: +SKIP
    >>> risk_scores = model.predict(X)  # doctest: +SKIP
    """

    def __init__(
        self,
        penalizer=0.0,
        l1_ratio=0.0,
        alpha=0.05,
        baseline_estimation_method="breslow",
    ):
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.baseline_estimation_method = baseline_estimation_method

    def _build_dataframe(self, X, y=None):
        """Build a DataFrame from X (and optionally y) for lifelines."""
        if hasattr(X, "columns"):
            df = pd.DataFrame(X.values if hasattr(X, "values") else X, columns=X.columns)
            self.feature_names_in_ = list(X.columns)
        else:
            X = np.asarray(X)
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=self.feature_names_in_)

        self.n_features_in_ = len(self.feature_names_in_)

        if y is not None:
            y = np.asarray(y)
            df["_event"] = y[:, 0].astype(bool)
            df["_duration"] = y[:, 1]

        return df

    def _build_predict_df(self, X):
        """Build a prediction DataFrame using stored feature names."""
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        return pd.DataFrame(X_arr, columns=self.feature_names_in_)

    def fit(self, X, y):
        """Fit the Cox PH model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples, 2)
            Survival target with columns ``[event, duration]``.

        Returns
        -------
        self
        """
        ll = _check_lifelines()
        df = self._build_dataframe(X, y)

        self.fitter_ = ll.CoxPHFitter(
            penalizer=self.penalizer,
            l1_ratio=self.l1_ratio,
            alpha=self.alpha,
            baseline_estimation_method=self.baseline_estimation_method,
        )
        self.fitter_.fit(df, duration_col="_duration", event_col="_event")
        return self

    def predict(self, X):
        """Predict log-partial hazard (risk scores).

        Higher values indicate higher risk (shorter expected survival).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
            Log-partial hazard values.
        """
        df = self._build_predict_df(X)
        return self.fitter_.predict_log_partial_hazard(df).values

    def score(self, X, y):
        """Compute Harrell's concordance index on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples, 2)
            Survival target with columns ``[event, duration]``.

        Returns
        -------
        c_index : float
            Concordance index in ``[0, 1]``.
        """
        ll = _check_lifelines()
        risk_scores = self.predict(X)
        y = np.asarray(y)
        event = y[:, 0].astype(bool)
        duration = y[:, 1]
        return ll.utils.concordance_index(duration, -risk_scores, event)

    def predict_survival_function(self, X, times=None):
        """Predict survival function for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        times : array-like or None, default=None
            Time points at which to evaluate the survival function.
            If ``None``, uses the unique event times from training.

        Returns
        -------
        survival : DataFrame
            Survival probabilities with times as index and samples
            as columns.
        """
        df = self._build_predict_df(X)
        if times is not None:
            return self.fitter_.predict_survival_function(df, times=times)
        return self.fitter_.predict_survival_function(df)

    def predict_median(self, X):
        """Predict median survival time for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        medians : ndarray of shape (n_samples,)
            Predicted median survival times.
        """
        df = self._build_predict_df(X)
        return self.fitter_.predict_median(df).values

    def __sklearn_tags__(self):
        try:
            from sklearn.utils._tags import Tags, TargetTags

            return Tags(
                estimator_type=None,
                target_tags=TargetTags(required=True),
                no_validation=True,
            )
        except ImportError:
            return {}
