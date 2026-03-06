"""Inner CV tuning report.

Provides :class:`InnerCVReport` for inspecting and diagnosing the
hyperparameter search that takes place inside each outer fold of a
nested cross-validation procedure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class InnerCVReport:
    """Diagnostics for the inner CV hyperparameter search of a single outer fold.

    Wraps the ``cv_results_`` dictionary produced by scikit-learn search
    objects (``GridSearchCV``, ``RandomizedSearchCV``, etc.) and exposes
    convenience methods for ranking configurations, estimating parameter
    importance, and examining score distributions.

    Parameters
    ----------
    cv_results : dict
        The ``cv_results_`` dictionary from the inner search object.
    outer_fold_idx : int
        Zero-based index of the outer fold this report belongs to.

    Attributes
    ----------
    cv_results_ : dict
        Raw ``cv_results_`` dictionary.
    outer_fold_idx : int
        Outer fold index.

    Examples
    --------
    >>> report = InnerCVReport(search.cv_results_, outer_fold_idx=0)  # doctest: +SKIP
    >>> report.top_k(3, metric="roc_auc")  # doctest: +SKIP

    See Also
    --------
    nestkit.inner.search.build_search : Construct the inner search object.
    """

    def __init__(self, cv_results: dict, outer_fold_idx: int):
        self.cv_results_ = cv_results
        self.outer_fold_idx = outer_fold_idx

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the full ``cv_results_`` dictionary to a DataFrame.

        Returns
        -------
        pandas.DataFrame
            One row per hyperparameter configuration with all columns
            from scikit-learn's ``cv_results_`` (parameters, mean/std
            scores, ranks, fit times, etc.).
        """
        return pd.DataFrame(self.cv_results_)

    def ranking(self, metric: str | None = None) -> pd.DataFrame:
        """Return all configurations ranked by mean inner-CV score.

        Sorts by the ``rank_test_<metric>`` column if it exists;
        otherwise falls back to descending ``mean_test_<metric>``.

        Parameters
        ----------
        metric : str or None, optional
            Metric name (e.g., ``"roc_auc"``).  If ``None``, uses the
            default ``"score"`` suffix from scikit-learn's single-metric
            results.

        Returns
        -------
        pandas.DataFrame
            Sorted configurations with all ``cv_results_`` columns.
        """
        df = self.to_dataframe()
        rank_col = self._rank_col(metric)
        if rank_col in df.columns:
            return df.sort_values(rank_col).reset_index(drop=True)
        score_col = self._score_col(metric)
        if score_col in df.columns:
            return df.sort_values(score_col, ascending=False).reset_index(drop=True)
        return df

    def top_k(self, k: int = 5, metric: str | None = None) -> pd.DataFrame:
        """Return the top *k* hyperparameter configurations.

        Parameters
        ----------
        k : int, default=5
            Number of top configurations to return.
        metric : str or None, optional
            Metric name for ranking.  See :meth:`ranking`.

        Returns
        -------
        pandas.DataFrame
            The *k* best-ranked rows from :meth:`ranking`.
        """
        return self.ranking(metric).head(k)

    def param_importance(self, metric: str | None = None) -> pd.DataFrame:
        """Estimate the marginal importance of each hyperparameter.

        Groups configurations by each ``param_*`` column and computes
        the variance of the group means (a simplified, fANOVA-inspired
        measure).  Higher variance indicates that the parameter has a
        larger effect on the inner-CV score.

        Parameters
        ----------
        metric : str or None, optional
            Metric name.  See :meth:`ranking`.

        Returns
        -------
        pandas.DataFrame
            Columns: ``parameter``, ``variance_explained``,
            ``n_unique``, ``relative_importance``.  Sorted by
            ``variance_explained`` in descending order.  Returns an
            empty DataFrame if the score column is not found.

        Notes
        -----
        This is a first-order marginal analysis and does not account
        for interactions between hyperparameters.  For a full fANOVA
        decomposition, consider dedicated tools such as ``fanova``.
        """
        df = self.to_dataframe()
        score_col = self._score_col(metric)
        if score_col not in df.columns:
            return pd.DataFrame()

        param_cols = [c for c in df.columns if c.startswith("param_")]
        rows = []
        for col in param_cols:
            grouped = df.groupby(col)[score_col]
            variance_explained = grouped.mean().var()
            rows.append(
                {
                    "parameter": col.replace("param_", ""),
                    "variance_explained": variance_explained
                    if not np.isnan(variance_explained)
                    else 0.0,
                    "n_unique": df[col].nunique(),
                }
            )

        result = pd.DataFrame(rows)
        if not result.empty:
            total = result["variance_explained"].sum()
            if total > 0:
                result["relative_importance"] = result["variance_explained"] / total
            else:
                result["relative_importance"] = 0.0
            result = result.sort_values("variance_explained", ascending=False)
        return result.reset_index(drop=True)

    def score_distribution(self, param: str, metric: str | None = None) -> pd.DataFrame:
        """Show mean inner-CV score as a function of a single hyperparameter.

        Useful for generating 1-D parameter-sweep plots.

        Parameters
        ----------
        param : str
            Hyperparameter name **without** the ``param_`` prefix.
        metric : str or None, optional
            Metric name.  See :meth:`ranking`.

        Returns
        -------
        pandas.DataFrame
            Columns include ``param_<param>``, the mean score column,
            and (if available) the standard deviation column.  Sorted
            by the parameter value.  Returns an empty DataFrame if
            the requested columns are not found.
        """
        df = self.to_dataframe()
        score_col = self._score_col(metric)
        param_col = f"param_{param}"
        if score_col not in df.columns or param_col not in df.columns:
            return pd.DataFrame()

        std_col = self._std_col(metric)
        cols = [param_col, score_col]
        if std_col in df.columns:
            cols.append(std_col)

        return df[cols].sort_values(param_col).reset_index(drop=True)

    def _rank_col(self, metric: str | None) -> str:
        if metric:
            return f"rank_test_{metric}"
        return "rank_test_score"

    def _score_col(self, metric: str | None) -> str:
        if metric:
            return f"mean_test_{metric}"
        return "mean_test_score"

    def _std_col(self, metric: str | None) -> str:
        if metric:
            return f"std_test_{metric}"
        return "std_test_score"
