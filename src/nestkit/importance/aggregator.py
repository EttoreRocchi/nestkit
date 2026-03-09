"""Feature importance aggregation across nested CV folds.

Provides :class:`FeatureImportanceAggregator`, which extracts per-fold
feature importances (via native model attributes or SHAP), aggregates
them into summary statistics, and quantifies selection stability using
the Nogueira et al. (2018) index.

See Also
--------
nestkit.importance.extractors : Low-level extraction helpers.
nestkit.importance.stability : Nogueira stability index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from nestkit._constants import _EPS
from nestkit.importance.extractors import compute_shap_importance, extract_model_importance
from nestkit.importance.stability import nogueira_stability_index
from nestkit.results._base import _BaseNestedCVResults


class FeatureImportanceAggregator:
    """Aggregate feature importances across nested CV outer folds.

    Extracts importance scores from each outer-fold estimator, optionally
    normalizes them, and computes summary statistics including mean,
    standard deviation, coefficient of variation, and rank-based
    diagnostics.  Also supports SHAP-based model-agnostic importances.

    Parameters
    ----------
    results : _BaseNestedCVResults
        Fitted nested CV results object.  Must have been produced with
        ``return_estimator=True`` so that per-fold estimators are
        available.
    method : {"auto", "model", "shap"}, default="auto"
        Importance extraction strategy.

        * ``"auto"`` / ``"model"`` -- use ``feature_importances_`` or
          ``coef_`` from the fitted estimator.
        * ``"shap"`` -- compute SHAP values on the outer test fold.
    feature_names : list[str] or None, optional
        Human-readable feature names.  If ``None``, inferred from
        ``results.feature_names_in_`` when available, otherwise
        ``feature_0``, ``feature_1``, etc.
    shap_type : {"tree", "kernel", "linear", "auto"}, default="auto"
        SHAP explainer backend.  Only used when *method* is ``"shap"``.
    normalize : bool, default=True
        If ``True``, absolute importances are rescaled to sum to 1
        within each fold.

    Attributes
    ----------
    importances_matrix_ : numpy.ndarray
        Shape ``(n_folds, n_features)``.  Set after :meth:`compute`.
    ranks_matrix_ : numpy.ndarray
        Shape ``(n_folds, n_features)``.  Rank of each feature per fold.
    summary_ : pandas.DataFrame
        Per-feature aggregated statistics.  Set after :meth:`compute`.

    Raises
    ------
    ValueError
        If *results* does not contain fitted estimators.

    Examples
    --------
    >>> agg = FeatureImportanceAggregator(results)  # doctest: +SKIP
    >>> agg.compute()  # doctest: +SKIP
    >>> agg.summary_.head()  # doctest: +SKIP

    See Also
    --------
    nestkit.importance.extractors.extract_model_importance
    nestkit.importance.extractors.compute_shap_importance
    nestkit.importance.stability.nogueira_stability_index
    """

    def __init__(
        self,
        results: _BaseNestedCVResults,
        method: str = "auto",
        feature_names: list[str] | None = None,
        shap_type: str = "auto",
        normalize: bool = True,
    ):
        if not results.has_fitted_estimators:
            raise ValueError(
                "Results do not contain fitted estimators. "
                "Re-run nested CV with return_estimator=True."
            )
        self.results = results
        self.method = method
        self.shap_type = shap_type
        self.normalize = normalize

        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(results, "feature_names_in_"):
            self.feature_names = results.feature_names_in_
        else:
            self.feature_names = None

    def compute(self, X=None, y=None) -> FeatureImportanceAggregator:
        """Extract and aggregate importances across all outer folds.

        Iterates over each outer-fold result, extracts importances using
        the configured method, optionally normalizes, and builds the
        ``importances_matrix_`` and ``summary_`` attributes.

        Parameters
        ----------
        X : array-like or None, optional
            Full feature matrix.  Required only when ``method="shap"``
            so that the outer test-fold subset can be sliced for the
            SHAP explainer.
        y : array-like or None, optional
            Target vector.  Currently unused; reserved for future
            supervised importance methods.

        Returns
        -------
        FeatureImportanceAggregator
            ``self``, to allow method chaining.

        Raises
        ------
        ValueError
            If ``method="shap"`` and *X* is ``None``.
        ValueError
            If *method* is not one of the recognised strategies.

        Notes
        -----
        When ``method="shap"``, SHAP values are computed **only on the
        outer test fold** of each split to avoid data leakage.

        See Also
        --------
        extract_model_importance : Model-native extraction.
        compute_shap_importance : SHAP-based extraction.
        """
        self.importances_per_fold_: list[np.ndarray] = []
        self.raw_importances_: list[np.ndarray] = []

        for fold_result in self.results.fold_results_:
            estimator = fold_result.fitted_estimator

            if self.method in ("auto", "model"):
                imp = extract_model_importance(estimator)
            elif self.method == "shap":
                if X is None:
                    raise ValueError("X is required for SHAP computation.")
                X_test = X[fold_result.test_indices]
                imp, raw = compute_shap_importance(estimator, X_test, self.shap_type)
                self.raw_importances_.append(raw)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            if self.normalize:
                abs_imp = np.abs(imp)
                total = abs_imp.sum()
                imp = abs_imp / (total + _EPS)

            self.importances_per_fold_.append(imp)

        self.importances_matrix_ = np.vstack(self.importances_per_fold_)
        self._compute_aggregates()
        self.is_computed_ = True
        return self

    def _compute_aggregates(self) -> None:
        M = self.importances_matrix_
        n_folds, n_features = M.shape
        names = self.feature_names or [f"feature_{i}" for i in range(n_features)]

        ranks = np.apply_along_axis(lambda x: rankdata(-x), 1, M)
        self.ranks_matrix_ = ranks

        self.summary_ = pd.DataFrame(
            {
                "feature": names,
                "mean_importance": M.mean(axis=0),
                "std_importance": M.std(axis=0, ddof=1) if n_folds > 1 else np.zeros(n_features),
                "median_importance": np.median(M, axis=0),
                "min_importance": M.min(axis=0),
                "max_importance": M.max(axis=0),
                "cv_importance": (
                    M.std(axis=0, ddof=1) / (M.mean(axis=0) + _EPS)
                    if n_folds > 1
                    else np.zeros(n_features)
                ),
                "mean_rank": ranks.mean(axis=0),
                "std_rank": ranks.std(axis=0, ddof=1) if n_folds > 1 else np.zeros(n_features),
            }
        )
        self.summary_ = self.summary_.sort_values("mean_importance", ascending=False).reset_index(
            drop=True
        )

    def stability_index(self, top_k: int = 10) -> float:
        """Compute the Nogueira et al. (2018) stability index for the top-k features.

        Measures how consistently the same features appear in the top-k
        set across outer folds.  A value of 1 indicates perfect
        agreement; 0 indicates random selection.

        Parameters
        ----------
        top_k : int, default=10
            Number of top features to consider per fold.

        Returns
        -------
        float
            Stability index in ``[-1, 1]``.

        See Also
        --------
        nestkit.importance.stability.nogueira_stability_index

        References
        ----------
        .. [1] Nogueira, S., Sechidis, K., and Brown, G. (2018).
               "On the Stability of Feature Selection Algorithms."
               *JMLR*, 18(174), 1--54.
        """
        return nogueira_stability_index(self.importances_matrix_, top_k)

    def consensus_features(
        self, criterion: str = "top_k", top_k: int = 10, min_frequency: float = 0.8
    ) -> list[str]:
        """Identify features that are consistently important across folds.

        Two selection strategies are available:

        * ``"top_k"`` -- return the *top_k* features by mean importance
          (from ``summary_``).
        * ``"frequency"`` -- return features that appear in the per-fold
          top-k set in at least *min_frequency* fraction of all folds.

        Parameters
        ----------
        criterion : {"top_k", "frequency"}, default="top_k"
            Selection strategy.
        top_k : int, default=10
            Number of top features per fold (used by both criteria).
        min_frequency : float, default=0.8
            Minimum fraction of folds in which a feature must appear in
            the top-k set.  Only used when ``criterion="frequency"``.

        Returns
        -------
        list[str]
            Feature names that satisfy the criterion.

        Raises
        ------
        ValueError
            If *criterion* is not recognised.

        Examples
        --------
        >>> agg.compute()  # doctest: +SKIP
        >>> agg.consensus_features("frequency", top_k=5, min_frequency=0.9)  # doctest: +SKIP
        """
        names = self.feature_names or [
            f"feature_{i}" for i in range(self.importances_matrix_.shape[1])
        ]

        if criterion == "top_k":
            return self.summary_.head(top_k)["feature"].tolist()

        if criterion == "frequency":
            n_folds, n_features = self.importances_matrix_.shape
            frequency = np.zeros(n_features)
            for i in range(n_folds):
                top_idx = np.argsort(-self.importances_matrix_[i])[:top_k]
                frequency[top_idx] += 1
            frequency /= n_folds
            mask = frequency >= min_frequency
            return [names[i] for i in range(n_features) if mask[i]]

        raise ValueError(f"Unknown criterion: {criterion}")

    def pairwise_rank_correlation(self) -> pd.DataFrame:
        """Compute Spearman rank correlation of feature importances between all fold pairs.

        High correlations indicate that the relative ordering of
        features is stable across outer folds.

        Returns
        -------
        pandas.DataFrame
            One row per fold pair with columns ``fold_i``, ``fold_j``,
            ``spearman_r``, and ``p_value``.

        Examples
        --------
        >>> agg.compute()  # doctest: +SKIP
        >>> agg.pairwise_rank_correlation()  # doctest: +SKIP
        """
        n_folds = self.ranks_matrix_.shape[0]
        rows = []
        for i in range(n_folds):
            for j in range(i + 1, n_folds):
                corr, p_value = spearmanr(self.ranks_matrix_[i], self.ranks_matrix_[j])
                rows.append(
                    {
                        "fold_i": i,
                        "fold_j": j,
                        "spearman_r": float(corr),
                        "p_value": float(p_value),
                    }
                )
        return pd.DataFrame(rows)
