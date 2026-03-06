"""Search strategy construction helpers.

Factory function for building scikit-learn (or scikit-optimize)
hyperparameter search objects used as the inner loop of nested
cross-validation.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def build_search(
    strategy: str | Any,
    estimator: BaseEstimator,
    param_grid: dict | list[dict],
    *,
    cv: Any = 5,
    scoring: Any = None,
    refit: Any = True,
    n_jobs: int | None = None,
    return_train_score: bool = False,
    error_score: str | float = "raise",
    verbose: int = 0,
    random_state: int | None = None,
) -> Any:
    """Build or clone a hyperparameter search object.

    When *strategy* is a string, a fresh ``GridSearchCV``,
    ``RandomizedSearchCV``, or ``BayesSearchCV`` is constructed with
    the given parameters.  When it is an existing search instance, it
    is cloned and its estimator is replaced.

    Parameters
    ----------
    strategy : str or BaseSearchCV instance
        Search strategy identifier.

        * ``"grid"`` -- exhaustive grid search via
          :class:`~sklearn.model_selection.GridSearchCV`.
        * ``"random"`` -- randomised search via
          :class:`~sklearn.model_selection.RandomizedSearchCV`.
        * ``"bayesian"`` -- Bayesian optimisation via
          :class:`skopt.BayesSearchCV` (requires *scikit-optimize*).
        * A ``BaseSearchCV`` instance -- cloned and reused with a fresh
          estimator.
    estimator : sklearn.base.BaseEstimator
        The estimator (or Pipeline) to tune.  A clone is made
        internally so the original is not mutated.
    param_grid : dict or list[dict]
        Parameter grid / distributions / search spaces passed to the
        underlying search object.
    cv : int or cross-validation generator, default=5
        Inner cross-validation splitting strategy.
    scoring : str, callable, dict, or None, default=None
        Scoring metric(s) for the inner search.  When a ``dict`` is
        passed and *refit* is ``True``, *refit* is automatically set
        to the first key.
    refit : bool or str, default=True
        Whether to refit the best estimator on the full inner
        training set.
    n_jobs : int or None, default=None
        Number of parallel jobs for the inner search.
    return_train_score : bool, default=False
        Whether to include training scores in ``cv_results_``.
    error_score : str or float, default="raise"
        Value assigned when a fit fails.
    verbose : int, default=0
        Verbosity level.  The search object receives
        ``max(0, verbose - 2)``.
    random_state : int or None, default=None
        Random seed for ``"random"`` and ``"bayesian"`` strategies.

    Returns
    -------
    sklearn.model_selection.BaseSearchCV
        Configured (but unfitted) search object.

    Raises
    ------
    ValueError
        If *strategy* is a string other than ``"grid"``, ``"random"``,
        or ``"bayesian"``.
    ImportError
        If ``strategy="bayesian"`` and *scikit-optimize* is not
        installed.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> search = build_search(  # doctest: +SKIP
    ...     "grid", SVC(), {"C": [0.1, 1, 10]}, cv=3, scoring="accuracy"
    ... )

    See Also
    --------
    nestkit.inner.tuning_report.InnerCVReport : Inspect inner search results.
    """
    if isinstance(strategy, str):
        # When scoring is a dict and refit=True, sklearn requires refit
        # to be a scorer key. Default to the first key.
        effective_refit = refit
        if isinstance(scoring, dict) and refit is True:
            effective_refit = next(iter(scoring))

        common = dict(
            estimator=clone(estimator),
            scoring=scoring,
            refit=effective_refit,
            cv=cv,
            n_jobs=n_jobs,
            return_train_score=return_train_score,
            error_score=error_score,
            verbose=max(0, verbose - 2),
        )

        if strategy == "grid":
            return GridSearchCV(param_grid=param_grid, **common)
        elif strategy == "random":
            return RandomizedSearchCV(
                param_distributions=param_grid,
                random_state=random_state,
                **common,
            )
        elif strategy == "bayesian":
            try:
                from skopt import BayesSearchCV

                return BayesSearchCV(
                    search_spaces=param_grid,
                    random_state=random_state,
                    **common,
                )
            except ImportError:
                raise ImportError(
                    "search_strategy='bayesian' requires scikit-optimize. "
                    "Install it with: pip install scikit-optimize"
                ) from None
        else:
            raise ValueError(
                f"search_strategy must be 'grid', 'random', 'bayesian' "
                f"or a BaseSearchCV instance, got '{strategy}'"
            )
    else:
        # Clone the provided search and override estimator
        search = clone(strategy)
        search.estimator = clone(estimator)
        return search
