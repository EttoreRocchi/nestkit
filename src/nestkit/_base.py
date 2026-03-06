"""Base class for nested cross-validation.

Provides the shared infrastructure for :class:`~nestkit.NestedCVClassifier`
and :class:`~nestkit.NestedCVRegressor`, including the outer-loop parallelism,
inner hyperparameter search dispatch, callback orchestration, and result
aggregation.
"""

from __future__ import annotations

import logging
import time
from abc import ABCMeta, abstractmethod
from typing import Any

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import check_cv
from sklearn.utils.validation import check_X_y

from nestkit.inner.search import build_search

logger = logging.getLogger("nestkit")


class _BaseNestedCV(BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for nested cross-validation estimators.

    This class is not intended to be instantiated directly. Use
    :class:`~nestkit.NestedCVClassifier` or :class:`~nestkit.NestedCVRegressor`
    instead.

    The nested CV procedure consists of four phases per outer fold:

    1. **Inner CV search**  -  hyperparameter tuning via grid or randomized
       search on the outer training set.
    2. **Post-inner processing**  -  task-specific operations such as
       probability calibration or prediction interval estimation
       (implemented by subclasses).
    3. **Refit**  -  the best hyperparameters are used to refit the estimator
       on the full outer training set.
    4. **Outer evaluation**  -  the refitted model is scored on the held-out
       outer test fold.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn compatible estimator that implements ``fit``.
        The estimator is cloned for each outer fold.
    param_grid : dict or list of dict
        Dictionary with parameter names (``str``) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries. Passed directly to the inner search strategy.
    search_strategy : {'grid', 'random', 'bayesian'}, default='grid'
        Hyperparameter search strategy. ``'grid'`` uses
        :class:`~sklearn.model_selection.GridSearchCV`; ``'random'`` uses
        :class:`~sklearn.model_selection.RandomizedSearchCV`;
        ``'bayesian'`` uses :class:`skopt.BayesSearchCV` (requires
        *scikit-optimize*).
    outer_cv : int, cross-validation generator, or iterable, default=5
        Determines the outer cross-validation splitting strategy.
        See :func:`sklearn.model_selection.check_cv` for accepted formats.
    inner_cv : int, cross-validation generator, or iterable, default=5
        Determines the inner cross-validation splitting strategy used
        for hyperparameter search.
    scoring : str, callable, list, tuple, or dict, default=None
        Scoring metric(s) for the inner search. If ``None``, the
        estimator's default scorer is used. See
        :func:`sklearn.metrics.get_scorer` for valid string values.
    refit : bool or str, default=True
        Whether to refit the best estimator on the full outer training
        set. If a string, it must match one of the scoring keys when
        multi-metric scoring is used.
    return_train_score : bool, default=False
        Whether to include training scores in the inner CV results.
    return_estimator : bool, default=True
        Whether to store the fitted estimator for each outer fold in
        the results.
    error_score : 'raise' or numeric, default='raise'
        Value to assign to the score if an error occurs during inner CV
        fitting. If ``'raise'``, the error is raised.
    n_jobs_outer : int or None, default=None
        Number of jobs for parallelizing the outer folds. ``None`` means
        sequential execution; ``-1`` uses all processors.
    n_jobs_inner : int or None, default=None
        Number of jobs for parallelizing the inner search within each
        outer fold.
    verbose : int, default=0
        Verbosity level. Values > 0 enable progressively more output
        from the inner search.
    random_state : int, RandomState instance, or None, default=None
        Controls randomness in the inner search (e.g., for
        ``RandomizedSearchCV``). Pass an int for reproducible results.
    callbacks : list of callback objects or None, default=None
        List of :class:`~nestkit.FoldCallback` objects that are notified
        at key points during the nested CV procedure.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs dispatched during parallel outer
        fold execution. See :class:`joblib.Parallel` for details.

    Notes
    -----
    Setting ``n_jobs_outer > 1`` together with ``n_jobs_inner > 1`` may
    cause thread oversubscription and degrade performance. As a rule of
    thumb, parallelize at one level only.

    See Also
    --------
    nestkit.NestedCVClassifier : Classification-specific nested CV.
    nestkit.NestedCVRegressor : Regression-specific nested CV.
    sklearn.model_selection.GridSearchCV : Inner search backend (grid).
    sklearn.model_selection.RandomizedSearchCV : Inner search backend (random).
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
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.search_strategy = search_strategy
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.scoring = scoring
        self.refit = refit
        self.return_train_score = return_train_score
        self.return_estimator = return_estimator
        self.error_score = error_score
        self.n_jobs_outer = n_jobs_outer
        self.n_jobs_inner = n_jobs_inner
        self.verbose = verbose
        self.random_state = random_state
        self.callbacks = callbacks
        self.pre_dispatch = pre_dispatch

    @abstractmethod
    def _post_inner_processing(self, search, X_train, y_train, groups_train, **fit_params) -> dict:
        """Called after inner search. Returns task-specific artifacts."""
        ...

    @abstractmethod
    def _evaluate_outer_fold(self, estimator, X_test, y_test, post_inner_artifacts) -> dict:
        """Evaluate best estimator on outer test fold."""
        ...

    @abstractmethod
    def _build_results_container(self) -> Any:
        """Return the appropriate results container class."""
        ...

    def fit(self, X, y, groups=None, **fit_params):
        """Run the full nested cross-validation procedure.

        For each outer fold the method performs: (1) inner hyperparameter
        search, (2) task-specific post-processing, (3) refit with the best
        parameters on the full outer training set, and (4) evaluation on
        the held-out outer test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. If a pandas DataFrame is passed, feature names
            and the original index are preserved in the results.
        y : array-like of shape (n_samples,)
            Target values.
        groups : array-like of shape (n_samples,) or None, default=None
            Group labels for the samples, used by group-aware CV
            splitters (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        **fit_params : dict
            Additional keyword arguments forwarded to the estimator's
            ``fit`` method in both the inner search and the final refit.

        Returns
        -------
        self
            The fitted nested CV estimator. Results are accessible via
            the :attr:`results_` attribute.

        Raises
        ------
        ValueError
            If ``X`` and ``y`` have incompatible shapes.
        """
        # DataFrame handling
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            self._original_index = X.index.copy()
            X = X.to_numpy()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
            self._original_index = None

        X, y = check_X_y(X, y, multi_output=True, allow_nd=True)

        outer_cv = check_cv(self.outer_cv, y, classifier=is_classifier(self.estimator))
        splits = list(outer_cv.split(X, y, groups))
        n_outer_folds = len(splits)

        # Initialize results container
        results_cls = self._build_results_container()
        self.results_ = results_cls(
            n_outer_folds=n_outer_folds,
            feature_names=self.feature_names_in_,
            original_index=self._original_index,
        )

        # Outer loop
        parallel = Parallel(
            n_jobs=self.n_jobs_outer,
            verbose=max(0, self.verbose - 1),
            pre_dispatch=self.pre_dispatch,
        )

        fold_results = parallel(
            delayed(self._fit_outer_fold)(
                clone(self.estimator),
                X,
                y,
                train_idx,
                test_idx,
                fold_idx,
                groups,
                **fit_params,
            )
            for fold_idx, (train_idx, test_idx) in enumerate(splits)
        )

        # Aggregate
        for fr in fold_results:
            self.results_.add_fold(fr)
        self.results_.finalize()

        # Callbacks
        if self.callbacks:
            for cb in self.callbacks:
                if hasattr(cb, "on_nested_cv_complete"):
                    cb.on_nested_cv_complete(self.results_)

        self.is_fitted_ = True
        return self

    def _fit_outer_fold(
        self, estimator, X, y, train_idx, test_idx, fold_idx, groups, **fit_params
    ):
        """Execute all four phases of a single outer fold.

        Parameters
        ----------
        estimator : estimator object
            Cloned base estimator for this fold.
        X : ndarray of shape (n_samples, n_features)
            Full feature matrix.
        y : ndarray of shape (n_samples,)
            Full target vector.
        train_idx : ndarray of shape (n_train,)
            Indices of the outer training set.
        test_idx : ndarray of shape (n_test,)
            Indices of the outer test set.
        fold_idx : int
            Zero-based index of the current outer fold.
        groups : ndarray or None
            Group labels for the full dataset.
        **fit_params : dict
            Additional keyword arguments forwarded to ``fit``.

        Returns
        -------
        fold_result : dataclass instance
            Task-specific fold result (e.g.,
            :class:`~nestkit.results.ClassifierOuterFoldResult`).
        """
        logger.info("Outer fold %d: starting", fold_idx)

        # Callbacks
        if self.callbacks:
            for cb in self.callbacks:
                if hasattr(cb, "on_outer_fold_start"):
                    cb.on_outer_fold_start(fold_idx, train_idx, test_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        groups_train = groups[train_idx] if groups is not None else None

        # Phase 1: Inner CV search
        t_start = time.time()
        search = build_search(
            strategy=self.search_strategy,
            estimator=estimator,
            param_grid=self.param_grid,
            cv=self.inner_cv,
            scoring=self.scoring,
            refit=self.refit,
            n_jobs=self.n_jobs_inner,
            return_train_score=self.return_train_score,
            error_score=self.error_score,
            verbose=self.verbose,
            random_state=self.random_state,
        )

        fit_kw = {}
        if groups_train is not None:
            fit_kw["groups"] = groups_train
        search.fit(X_train, y_train, **fit_kw, **fit_params)

        fit_time = time.time() - t_start
        logger.info(
            "Outer fold %d: inner search complete (%.1fs), best_score=%.4f",
            fold_idx,
            fit_time,
            search.best_score_,
        )

        # Callbacks
        if self.callbacks:
            for cb in self.callbacks:
                if hasattr(cb, "on_inner_search_complete"):
                    cb.on_inner_search_complete(fold_idx, search)

        # Post-inner processing (calibration, thresholding, etc.)
        artifacts = self._post_inner_processing(
            search, X_train, y_train, groups_train, **fit_params
        )

        if self.callbacks:
            for cb in self.callbacks:
                if hasattr(cb, "on_post_processing_complete"):
                    cb.on_post_processing_complete(fold_idx, artifacts)

        # Phase 4: Refit on full outer train
        best_params = search.best_params_
        final_estimator = clone(self.estimator).set_params(**best_params)
        final_estimator.fit(X_train, y_train, **fit_params)

        # Evaluate on outer test
        t_score = time.time()
        eval_result = self._evaluate_outer_fold(final_estimator, X_test, y_test, artifacts)
        score_time = time.time() - t_score

        # Build fold result
        fold_result = self._build_fold_result(
            fold_idx=fold_idx,
            train_idx=train_idx,
            test_idx=test_idx,
            best_params=best_params,
            best_inner_score=float(search.best_score_),
            inner_cv_results=search.cv_results_,
            fit_time=fit_time,
            score_time=score_time,
            estimator=final_estimator if self.return_estimator else None,
            artifacts=artifacts,
            eval_result=eval_result,
        )

        if self.callbacks:
            for cb in self.callbacks:
                if hasattr(cb, "on_outer_fold_complete"):
                    cb.on_outer_fold_complete(fold_idx, fold_result)

        logger.info("Outer fold %d: complete", fold_idx)
        return fold_result

    @abstractmethod
    def _build_fold_result(self, **kwargs) -> Any:
        """Build the task-specific fold result dataclass."""
        ...

    # sklearn compatibility

    def __sklearn_tags__(self):
        try:
            from sklearn.utils._tags import Tags, TargetTags

            return Tags(
                estimator_type=None,
                target_tags=TargetTags(required=True),
                no_validation=False,
            )
        except ImportError:
            return {"estimator_type": None, "no_validation": False}

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "is_fitted_") and self.is_fitted_
