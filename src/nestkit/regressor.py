"""Nested cross-validation estimator for regression tasks.

Extends :class:`~nestkit._base._BaseNestedCV` with optional residual-based
prediction intervals computed from inner out-of-fold residuals.
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import check_cv

from nestkit._base import _BaseNestedCV
from nestkit.results.regressor_results import RegressorOuterFoldResult, RegressorResults

logger = logging.getLogger("nestkit")


class NestedCVRegressor(_BaseNestedCV):
    """Nested cross-validation for regression tasks.

    Extends :class:`~nestkit._base._BaseNestedCV` with support for
    residual-based prediction intervals. When ``prediction_intervals=True``,
    inner out-of-fold residuals are collected and their quantiles (with
    finite-sample correction) are used to construct prediction intervals
    on the outer test set.

    .. note::

       The residuals are collected from OOF models fitted with the best
       hyperparameters, but the final model is refitted on the full outer
       training set. The residual distribution may therefore not perfectly
       match the final model's errors. These intervals are approximate
       and do not carry formal conformal coverage guarantees.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn compatible regressor that implements ``fit``
        and ``predict``. Cloned for each outer fold.
    param_grid : dict or list of dict
        Hyperparameter search space.
    search_strategy : {'grid', 'random', 'bayesian'}, default='grid'
        Inner hyperparameter search strategy.
    outer_cv : int, cross-validation generator, or iterable, default=5
        Outer cross-validation splitting strategy.
    inner_cv : int, cross-validation generator, or iterable, default=5
        Inner cross-validation splitting strategy.
    scoring : str, callable, list, tuple, or dict, default=None
        Scoring metric(s) for the inner search.
    refit : bool or str, default=True
        Whether to refit on the full outer training set.
    return_train_score : bool, default=False
        Whether to include training scores in inner CV results.
    return_estimator : bool, default=True
        Whether to store fitted estimators per outer fold.
    error_score : 'raise' or numeric, default='raise'
        Value assigned on inner CV fitting errors.
    n_jobs_outer : int or None, default=None
        Number of parallel jobs for outer folds.
    n_jobs_inner : int or None, default=None
        Number of parallel jobs for inner search.
    verbose : int, default=0
        Verbosity level.
    random_state : int, RandomState instance, or None, default=None
        Random state for reproducibility.
    callbacks : list of callback objects or None, default=None
        :class:`~nestkit.FoldCallback` instances for monitoring.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls job dispatch for parallel execution.
    prediction_intervals : bool, default=False
        Whether to compute residual-based prediction intervals using
        inner out-of-fold residuals. When enabled, the results contain
        ``prediction_interval_lower`` and ``prediction_interval_upper``
        per outer fold.
    confidence_level : float, default=0.95
        Confidence level for prediction intervals (e.g., 0.95 for 95%
        intervals). Only used when ``prediction_intervals=True``.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import Ridge
    >>> from nestkit import NestedCVRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> ncv = NestedCVRegressor(
    ...     estimator=Ridge(),
    ...     param_grid={"alpha": [0.01, 0.1, 1.0, 10.0]},
    ...     outer_cv=5, inner_cv=3,
    ...     prediction_intervals=True,
    ...     random_state=42,
    ... )
    >>> ncv.fit(X, y)  # doctest: +SKIP
    >>> print(ncv.results_.summary_default_)  # doctest: +SKIP

    See Also
    --------
    nestkit.NestedCVClassifier : Classification-specific nested CV.
    """

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        search_strategy="grid",
        outer_cv=5,
        inner_cv=5,
        scoring=None,
        refit=True,
        return_train_score=False,
        return_estimator=True,
        error_score="raise",
        n_jobs_outer=None,
        n_jobs_inner=None,
        verbose=0,
        random_state=None,
        callbacks=None,
        pre_dispatch="2*n_jobs",
        prediction_intervals=False,
        confidence_level=0.95,
    ):
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            search_strategy=search_strategy,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            scoring=scoring,
            refit=refit,
            return_train_score=return_train_score,
            return_estimator=return_estimator,
            error_score=error_score,
            n_jobs_outer=n_jobs_outer,
            n_jobs_inner=n_jobs_inner,
            verbose=verbose,
            random_state=random_state,
            callbacks=callbacks,
            pre_dispatch=pre_dispatch,
        )
        self.prediction_intervals = prediction_intervals
        self.confidence_level = confidence_level

    def _build_results_container(self):
        return RegressorResults

    def _post_inner_processing(self, search, X_train, y_train, groups_train, **fit_params):
        """Collect residuals for prediction intervals if enabled."""
        artifacts = {
            "residual_quantiles": None,
        }

        if not self.prediction_intervals:
            return artifacts

        # Collect inner OOF residuals for prediction intervals
        cal_cv = check_cv(self.inner_cv, y_train, classifier=False)
        best_params = search.best_params_
        base_estimator = clone(self.estimator).set_params(**best_params)

        all_residuals = []
        for inner_train_idx, inner_val_idx in cal_cv.split(X_train, y_train, groups_train):
            est_j = clone(base_estimator)
            est_j.fit(X_train[inner_train_idx], y_train[inner_train_idx], **fit_params)
            preds = est_j.predict(X_train[inner_val_idx])
            residuals = y_train[inner_val_idx] - preds
            all_residuals.extend(residuals)

        all_residuals = np.array(all_residuals)
        n_cal = len(all_residuals)
        alpha = 1 - self.confidence_level

        # Finite-sample corrected quantile levels (conformal-style)
        # Ref: Vovk et al., Algorithmic Learning in a Random World, 2005
        q_lo = alpha / 2
        q_hi = 1 - alpha / 2
        if n_cal > 0:
            q_lo = max(0.0, np.floor((alpha / 2) * (n_cal + 1)) / n_cal)
            q_hi = min(1.0, np.ceil((1 - alpha / 2) * (n_cal + 1)) / n_cal)

        artifacts["residual_quantiles"] = (
            float(np.quantile(all_residuals, q_lo)),
            float(np.quantile(all_residuals, q_hi)),
        )

        return artifacts

    def _evaluate_outer_fold(self, estimator, X_test, y_test, artifacts):
        """Evaluate on outer test set."""
        y_pred = estimator.predict(X_test)
        residuals = y_test - y_pred

        scores = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }
        with contextlib.suppress(Exception):
            scores["mape"] = mean_absolute_percentage_error(y_test, y_pred)

        y_pred_lower = None
        y_pred_upper = None
        coverage = None

        if artifacts.get("residual_quantiles") is not None:
            q_lo, q_hi = artifacts["residual_quantiles"]
            y_pred_lower = y_pred + q_lo
            y_pred_upper = y_pred + q_hi
            coverage = float(np.mean((y_test >= y_pred_lower) & (y_test <= y_pred_upper)))

        return {
            "y_true": y_test,
            "y_pred": y_pred,
            "residuals": residuals,
            "scores": scores,
            "y_pred_lower": y_pred_lower,
            "y_pred_upper": y_pred_upper,
            "coverage": coverage,
        }

    def _build_fold_result(self, **kwargs):
        eval_result = kwargs.pop("eval_result")
        kwargs.pop("artifacts")

        return RegressorOuterFoldResult(
            fold_idx=kwargs["fold_idx"],
            train_indices=kwargs["train_idx"],
            test_indices=kwargs["test_idx"],
            best_params=kwargs["best_params"],
            best_inner_score=kwargs["best_inner_score"],
            inner_cv_results=kwargs["inner_cv_results"],
            fit_time=kwargs["fit_time"],
            score_time=kwargs["score_time"],
            fitted_estimator=kwargs["estimator"],
            y_true=eval_result["y_true"],
            y_pred=eval_result["y_pred"],
            outer_scores=eval_result["scores"],
            residuals=eval_result["residuals"],
            prediction_interval_lower=eval_result.get("y_pred_lower"),
            prediction_interval_upper=eval_result.get("y_pred_upper"),
            coverage=eval_result.get("coverage"),
        )
