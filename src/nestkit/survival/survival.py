"""Nested cross-validation estimator for survival analysis.

Extends :class:`~nestkit._base._BaseNestedCV` with support for
Cox proportional hazards models from lifelines, providing Harrell's
and Uno's concordance indices, integrated Brier score, and
coefficient stability analysis across outer folds.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold

from nestkit._base import _BaseNestedCV
from nestkit._validation import validate_survival_target
from nestkit.results.survival_results import SurvivalOuterFoldResult, SurvivalResults
from nestkit.survival._scoring import (
    _compute_ibs,
    _compute_uno_c_index,
    _truncate_ibs_eval_times,
    uno_c_index_scorer,
)
from nestkit.survival._target import _normalize_survival_target

_IBS_QUANTILES = np.linspace(0.1, 0.9, 9)
_MIN_UNCENSORED_FOR_IBS = 5


def _check_ibs_eval_times(times):
    """Validate and sort a user-supplied IBS evaluation grid.

    The integrated Brier score is integrated over this grid with the
    trapezoidal rule, so the times must be sorted; they must also be
    finite and positive to be valid follow-up times.
    """
    times = np.asarray(times, dtype=np.float64).ravel()
    if times.size == 0:
        raise ValueError("ibs_eval_times must contain at least one time point")
    if not np.all(np.isfinite(times)):
        raise ValueError("ibs_eval_times must contain only finite values")
    if np.any(times <= 0):
        raise ValueError("ibs_eval_times must contain only positive values")
    return np.sort(times)


def _default_ibs_grid(y):
    """Quantile grid of the uncensored event times in *y*, or ``None``.

    Returns ``None`` when there are too few observed events for the
    quantiles to be meaningful, in which case IBS is skipped.
    """
    event = y[:, 0].astype(bool)
    uncensored = y[event, 1]
    if uncensored.shape[0] < _MIN_UNCENSORED_FOR_IBS:
        return None
    return np.quantile(uncensored, _IBS_QUANTILES)


def _split_pipeline(estimator):
    """Return ``(preprocessor, final_step)``; ``preprocessor`` is None if absent."""
    if hasattr(estimator, "steps"):
        return (estimator[:-1] if len(estimator.steps) > 1 else None), estimator[-1]
    return None, estimator


class _SurvivalStratifiedKFold:
    """Stratified K-Fold that extracts the event column from 2-column y.

    Wraps :class:`~sklearn.model_selection.StratifiedKFold` to handle
    survival targets of shape ``(n_samples, 2)`` where column 0 is
    the event indicator.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self._inner = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(self, X, y=None, groups=None):
        if y is not None and hasattr(y, "ndim") and y.ndim == 2:
            event = y[:, 0].astype(int)
        else:
            event = y
        yield from self._inner.split(X, event, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._inner.get_n_splits(X, y, groups)


class NestedCVSurvival(_BaseNestedCV):
    """Nested cross-validation for survival analysis.

    Wraps a scikit-learn compatible survival estimator (typically
    :class:`~nestkit.survival.CoxPHWrapper`) in a nested CV loop.
    The outer folds are stratified by event indicator to maintain
    representative censoring ratios.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn compatible survival estimator that implements
        ``fit(X, y)`` and ``predict(X)`` where ``y`` is a 2-column
        array ``[event, duration]``. Use
        :class:`~nestkit.survival.CoxPHWrapper` for lifelines'
        ``CoxPHFitter``. A :class:`~sklearn.pipeline.Pipeline` ending
        in such an estimator is also accepted (e.g. with an imputer as
        first step, since ``CoxPHFitter`` does not accept NaN);
        coefficients and the integrated Brier score are then taken
        from its final step.
    param_grid : dict or list of dict
        Hyperparameter search space.
    search_strategy : {'grid', 'random', 'bayesian'}, default='grid'
        Inner hyperparameter search strategy.
    outer_cv : int, cross-validation generator, or iterable, default=5
        Outer cross-validation splitting strategy.  If an int,
        :class:`StratifiedKFold` on the event indicator is used.
    inner_cv : int, cross-validation generator, or iterable, default=5
        Inner cross-validation splitting strategy.
    scoring : str, callable, or None, default=None
        Scoring metric for the inner search.  If ``None``, Uno's
        concordance index is used.
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
    ibs_eval_times : array-like or None, default=None
        Explicit time points for integrated Brier score computation.
        When given, the same grid is used for every outer fold and
        ``ibs_time_grid`` is ignored.  If ``None``, the grid is built
        from 9 equally-spaced quantiles of the uncensored event times,
        as selected by ``ibs_time_grid``.
    ibs_time_grid : {'per_fold', 'global'}, default='per_fold'
        How to derive the default IBS evaluation grid when
        ``ibs_eval_times`` is ``None``.

        - ``'per_fold'``: each outer fold derives its grid from the
          quantiles of the uncensored event times of *its own training
          rows*.  No information from the test fold enters the
          evaluation, consistent with the leakage-free guarantee
          documented in :ref:`architecture`.  The integration domains
          differ slightly between folds, so the mean reported in
          ``summary_default_`` averages Brier scores over close but
          not identical domains.
        - ``'global'``: a single grid derived from the uncensored
          event times of the whole dataset.  Every fold is integrated
          over the same domain, which makes per-fold scores directly
          comparable, at the cost of letting the test folds' event
          times influence *where* the score is evaluated.  The grid
          never enters model fitting.

        Pass ``ibs_eval_times`` explicitly for a domain fixed a priori,
        which avoids the trade-off entirely.

        Whichever grid is used, evaluation times beyond the horizon
        where the training censoring distribution reaches zero are
        dropped with a warning, since the
        inverse-probability-of-censoring weights are undefined there.

    Examples
    --------
    >>> import numpy as np
    >>> from nestkit.survival import NestedCVSurvival, CoxPHWrapper, make_survival_target
    >>> X = np.random.randn(200, 10)
    >>> event = np.random.binomial(1, 0.7, 200)
    >>> duration = np.random.exponential(10, 200)
    >>> y = make_survival_target(event, duration)
    >>> ncv = NestedCVSurvival(
    ...     estimator=CoxPHWrapper(),
    ...     param_grid={"penalizer": [0.01, 0.1, 1.0]},
    ...     outer_cv=5, inner_cv=3,
    ...     random_state=42,
    ... )
    >>> ncv.fit(X, y)  # doctest: +SKIP
    >>> print(ncv.results_.summary_default_)  # doctest: +SKIP

    See Also
    --------
    nestkit.NestedCVClassifier : Classification-specific nested CV.
    nestkit.NestedCVRegressor : Regression-specific nested CV.
    nestkit.survival.CoxPHWrapper : sklearn wrapper for CoxPHFitter.
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
        ibs_eval_times=None,
        ibs_time_grid="per_fold",
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
        self.ibs_eval_times = ibs_eval_times
        self.ibs_time_grid = ibs_time_grid

    def fit(self, X, y, groups=None, **fit_params):
        """Run nested cross-validation for survival analysis.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. May contain ``NaN`` if the estimator handles
            them, e.g. a Pipeline starting with an imputer (see
            :ref:`missing-values`).
        y : array-like
            Survival target.  Accepted formats:

            - 2-column ndarray ``[event, duration]``
            - Structured array with ``'event'`` and ``'duration'`` fields
            - DataFrame with ``'event'`` and ``'duration'`` columns

            Use :func:`~nestkit.survival.make_survival_target` to
            construct from separate arrays.
        groups : array-like of shape (n_samples,) or None, default=None
            Group labels for group-aware CV splitters.
        **fit_params : dict
            Additional keyword arguments forwarded to the estimator's
            ``fit`` method.

        Returns
        -------
        self
        """
        # Normalize and validate target
        y = _normalize_survival_target(y)
        validate_survival_target(y)

        if self.ibs_time_grid not in ("per_fold", "global"):
            raise ValueError(
                f"ibs_time_grid must be 'per_fold' or 'global', got {self.ibs_time_grid!r}"
            )

        saved_outer_cv = self.outer_cv
        saved_inner_cv = self.inner_cv
        saved_scoring = self.scoring

        try:
            if isinstance(self.outer_cv, int):
                self.outer_cv = _SurvivalStratifiedKFold(
                    n_splits=self.outer_cv,
                    shuffle=True,
                    random_state=self.random_state,
                )
            if isinstance(self.inner_cv, int):
                self.inner_cv = _SurvivalStratifiedKFold(
                    n_splits=self.inner_cv,
                    shuffle=True,
                    random_state=self.random_state,
                )

            if self.scoring is None:
                self.scoring = uno_c_index_scorer

            if self.ibs_eval_times is not None:
                self._ibs_eval_times_ = _check_ibs_eval_times(self.ibs_eval_times)
            elif self.ibs_time_grid == "global":
                self._ibs_eval_times_ = _default_ibs_grid(y)
            else:
                self._ibs_eval_times_ = None

            n_uncensored = int(np.count_nonzero(y[:, 0]))
            if self.ibs_eval_times is None and n_uncensored < _MIN_UNCENSORED_FOR_IBS:
                warnings.warn(
                    f"Too few uncensored events for IBS computation ({n_uncensored}). "
                    "IBS will be skipped.",
                    UserWarning,
                    stacklevel=2,
                )

            return super().fit(X, y, groups=groups, **fit_params)
        finally:
            self.outer_cv = saved_outer_cv
            self.inner_cv = saved_inner_cv
            self.scoring = saved_scoring

    def _build_results_container(self):
        return SurvivalResults

    def _remap_coefficients(self, coef_dict, feature_names=None):
        """Remap generic wrapper names to real feature names.

        The base class converts DataFrames to numpy before passing to
        the estimator, so ``CoxPHWrapper`` internally uses generic
        names (``feature_0``, ...).  This method restores the names in
        *feature_names* (default: ``self.feature_names_in_``).
        """
        names = self.feature_names_in_ if feature_names is None else feature_names
        if coef_dict is None or not names:
            return coef_dict
        name_map = {f"feature_{i}": name for i, name in enumerate(names)}
        return {name_map.get(k, k): v for k, v in coef_dict.items()}

    def _extract_coefficients(self, estimator):
        """Return the Cox coefficients keyed by feature name, or ``None``.

        For a Pipeline, coefficients are read from the final step and
        named after the output features of the preprocessing steps, so
        that columns dropped or added by them (e.g. by an imputer)
        do not shift the names.
        """
        preprocessor, final = _split_pipeline(estimator)
        if not hasattr(final, "fitter_"):
            return None
        coef_dict = final.fitter_.params_.to_dict()
        names = self.feature_names_in_
        if preprocessor is not None:
            try:
                names = list(preprocessor.get_feature_names_out(names))
            except (AttributeError, ValueError):
                if len(names) != len(coef_dict):
                    return coef_dict
        return self._remap_coefficients(coef_dict, names)

    def _post_inner_processing(self, search, X_train, y_train, groups_train, **fit_params):
        """Extract coefficients and store training data for outer metrics."""
        coefficients = self._extract_coefficients(search.best_estimator_)

        if self.ibs_eval_times is not None or self.ibs_time_grid == "global":
            fold_eval_times = self._ibs_eval_times_
        else:
            fold_eval_times = _default_ibs_grid(y_train)
            if fold_eval_times is None:
                warnings.warn(
                    "Too few uncensored events in one outer fold's training set "
                    f"(< {_MIN_UNCENSORED_FOR_IBS}) to build an IBS evaluation grid. "
                    "IBS is skipped for that fold.",
                    UserWarning,
                    stacklevel=2,
                )

        return {
            "coefficients": coefficients,
            "train_event": y_train[:, 0],
            "train_duration": y_train[:, 1],
            "ibs_eval_times": fold_eval_times,
        }

    def _evaluate_outer_fold(self, estimator, X_test, y_test, artifacts):
        """Evaluate on outer test set with survival-specific metrics."""
        from nestkit.survival._wrapper import _check_lifelines

        ll = _check_lifelines()
        risk_scores = estimator.predict(X_test)
        event = y_test[:, 0].astype(bool)
        duration = y_test[:, 1]

        train_event = artifacts["train_event"]
        train_duration = artifacts["train_duration"]

        c_index = ll.utils.concordance_index(duration, -risk_scores, event)

        uno_c = _compute_uno_c_index(train_event, train_duration, event, duration, risk_scores)

        scores = {
            "concordance_index": c_index,
            "uno_c_index": uno_c,
        }

        preprocessor, final = _split_pipeline(estimator)
        eval_times = artifacts.get("ibs_eval_times")
        used_eval_times = None
        if eval_times is not None and hasattr(final, "predict_survival_function"):
            eval_times = _truncate_ibs_eval_times(eval_times, train_event, train_duration)

            if eval_times.shape[0] == 0:
                warnings.warn(
                    "IBS skipped for one outer fold: every evaluation time lies "
                    "beyond the horizon where the training censoring distribution "
                    "is positive.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                try:
                    X_sf = X_test if preprocessor is None else preprocessor.transform(X_test)
                    sf = final.predict_survival_function(X_sf, times=eval_times)
                    sf_array = np.asarray(sf, dtype=np.float64)

                    ibs = _compute_ibs(
                        train_event,
                        train_duration,
                        event,
                        duration,
                        sf_array,
                        eval_times,
                    )
                    scores["integrated_brier_score"] = ibs
                    used_eval_times = eval_times
                except Exception as e:
                    warnings.warn(
                        f"IBS computation failed for one outer fold and was skipped: {e!r}",
                        UserWarning,
                        stacklevel=2,
                    )

        coefficients = self._extract_coefficients(estimator)

        return {
            "y_event": event,
            "y_duration": duration,
            "risk_scores": risk_scores,
            "scores": scores,
            "coefficients": coefficients,
            "ibs_eval_times": used_eval_times,
        }

    def _build_fold_result(self, **kwargs):
        eval_result = kwargs.pop("eval_result")
        kwargs.pop("artifacts")

        return SurvivalOuterFoldResult(
            fold_idx=kwargs["fold_idx"],
            train_indices=kwargs["train_idx"],
            test_indices=kwargs["test_idx"],
            best_params=kwargs["best_params"],
            best_inner_score=kwargs["best_inner_score"],
            inner_cv_results=kwargs["inner_cv_results"],
            fit_time=kwargs["fit_time"],
            score_time=kwargs["score_time"],
            fitted_estimator=kwargs["estimator"],
            y_event=eval_result["y_event"],
            y_duration=eval_result["y_duration"],
            risk_scores=eval_result["risk_scores"],
            outer_scores=eval_result["scores"],
            coefficients=eval_result.get("coefficients"),
            ibs_eval_times=eval_result.get("ibs_eval_times"),
        )
