"""Per-fold feature importance extraction.

Provides helpers to extract importance scores from scikit-learn
estimators (including Pipelines) using either native model attributes
(``feature_importances_``, ``coef_``) or SHAP explainers.

Notes
-----
When using ``compute_shap_importance`` with ``shap_type="kernel"``
(or ``"auto"`` for models without tree or linear attributes), the
``KernelExplainer`` is used.  This explainer is **very slow** for
large datasets because it evaluates the model on many perturbed
samples.  Consider subsampling or using a model-specific explainer
when possible.

See Also
--------
nestkit.importance.aggregator.FeatureImportanceAggregator
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator


def extract_model_importance(estimator: BaseEstimator) -> np.ndarray:
    """Extract feature importances from model-native attributes.

    Supports any scikit-learn estimator that exposes
    ``feature_importances_`` (tree-based models) or ``coef_`` (linear
    models).  If the estimator is a ``Pipeline``, the final step is
    unwrapped automatically.

    For linear models with a 2-D ``coef_`` (multi-class), the absolute
    coefficients are averaged across classes.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        A fitted estimator or Pipeline.

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(n_features,)`` with non-negative
        importance scores.

    Raises
    ------
    AttributeError
        If the (unwrapped) estimator has neither
        ``feature_importances_`` nor ``coef_``.

    See Also
    --------
    compute_shap_importance : Model-agnostic alternative.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> rf = RandomForestClassifier().fit(X_train, y_train)  # doctest: +SKIP
    >>> imp = extract_model_importance(rf)  # doctest: +SKIP
    """
    est = _unwrap_pipeline(estimator)

    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_)
    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        if coef.ndim == 2:
            return np.mean(np.abs(coef), axis=0)
        return np.abs(coef)

    raise AttributeError(
        f"{type(est).__name__} has neither feature_importances_ nor coef_. "
        f"Use method='shap' for model-agnostic importance."
    )


def compute_shap_importance(
    estimator: BaseEstimator,
    X_test: np.ndarray,
    shap_type: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values on an outer test fold.

    Selects the appropriate SHAP explainer based on *shap_type* (or
    auto-detected from the estimator) and computes SHAP values for each
    sample in *X_test*.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        A fitted estimator or Pipeline.
    X_test : numpy.ndarray
        Test-fold feature matrix, shape ``(n_samples, n_features)``.
    shap_type : {"tree", "kernel", "linear", "auto"}, default="auto"
        Explainer backend to use.  ``"auto"`` selects ``TreeExplainer``
        for tree-based models, ``LinearExplainer`` for linear models,
        and ``KernelExplainer`` otherwise.

    Returns
    -------
    mean_abs_shap : numpy.ndarray
        Mean absolute SHAP value per feature, shape ``(n_features,)``.
    raw_shap_values : numpy.ndarray
        Raw SHAP values, shape ``(n_samples, n_features)``.

    Raises
    ------
    ImportError
        If the ``shap`` package is not installed.

    Notes
    -----
    ``KernelExplainer`` (used when ``shap_type="kernel"`` or when
    auto-detection falls back) is **very slow** because it evaluates the
    model on ``O(n_features * n_background)`` perturbed samples.
    Consider tree- or linear-specific explainers when possible.

    For binary classifiers that return a list of two SHAP arrays, only
    the positive-class array (index 1) is used.

    See Also
    --------
    extract_model_importance : Faster model-native alternative.

    Examples
    --------
    >>> mean_shap, raw = compute_shap_importance(  # doctest: +SKIP
    ...     pipeline, X_test, shap_type="tree"
    ... )
    """
    import shap

    est = _unwrap_pipeline(estimator)
    explainer_cls = _resolve_shap_explainer(est, shap_type)

    if explainer_cls == shap.TreeExplainer:
        explainer = explainer_cls(est)
    elif explainer_cls == shap.LinearExplainer:
        explainer = explainer_cls(est, X_test)
    else:
        background = shap.kmeans(X_test, min(50, len(X_test)))
        explainer = explainer_cls(est.predict_proba, background)

    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    return mean_abs_shap, shap_values


def _unwrap_pipeline(estimator: BaseEstimator) -> BaseEstimator:
    """Extract the final estimator from a Pipeline."""
    if hasattr(estimator, "steps"):
        return estimator[-1]
    return estimator


def _resolve_shap_explainer(estimator, shap_type: str = "auto"):
    """Determine the appropriate SHAP explainer."""
    import shap

    if shap_type != "auto":
        return {
            "tree": shap.TreeExplainer,
            "kernel": shap.KernelExplainer,
            "linear": shap.LinearExplainer,
        }[shap_type]

    if hasattr(estimator, "feature_importances_"):
        return shap.TreeExplainer
    if hasattr(estimator, "coef_"):
        return shap.LinearExplainer
    return shap.KernelExplainer
