"""Base results container for nested cross-validation.

Provides the abstract base class ``_BaseNestedCVResults`` from which
task-specific results containers (``ClassifierResults``,
``RegressorResults``) inherit.  Handles fold aggregation, summary
statistics with Nadeau--Bengio corrected confidence intervals, and
serialisation to dict / DataFrame / JSON / LaTeX.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


class _BaseNestedCVResults(ABC):
    """Base class for nested CV results aggregation.

    Collects per-fold results produced during a nested cross-validation
    run and provides methods for computing aggregate statistics,
    exporting results, and inspecting generalisation behaviour.

    Subclasses must implement :meth:`finalize` to compute task-specific
    aggregates (e.g., confusion matrices for classification, residual
    statistics for regression).

    Parameters
    ----------
    n_outer_folds : int
        Number of outer cross-validation folds.
    feature_names : list of str or None, optional
        Feature names used during training, stored for downstream
        reporting.  ``None`` if not available.
    original_index : array-like or None, optional
        Original sample indices (e.g., a pandas Index) used to align
        out-of-fold predictions back to the source dataset.

    Attributes
    ----------
    n_outer_folds_ : int
        Number of outer folds (set at construction).
    feature_names_in_ : list of str or None
        Feature names, if provided.
    fold_results_ : list
        Accumulated per-fold result objects (populated via
        :meth:`add_fold`).

    See Also
    --------
    nestkit.results.classifier_results.ClassifierResults :
        Concrete subclass for classification tasks.
    nestkit.results.regressor_results.RegressorResults :
        Concrete subclass for regression tasks.
    """

    def __init__(
        self,
        n_outer_folds: int,
        feature_names: list[str] | None = None,
        original_index: Any | None = None,
    ):
        self.n_outer_folds_ = n_outer_folds
        self.feature_names_in_ = feature_names
        self._original_index = original_index
        self.fold_results_: list = []
        self._finalized = False

    def add_fold(self, fold_result: Any) -> None:
        """Append a single outer-fold result to the collection.

        Parameters
        ----------
        fold_result : ClassifierOuterFoldResult or RegressorOuterFoldResult
            The result object produced by evaluating one outer fold.
        """
        self.fold_results_.append(fold_result)

    @abstractmethod
    def finalize(self) -> None:
        """Compute aggregate statistics after all folds are added.

        Must be called exactly once, after all ``n_outer_folds`` fold
        results have been added via :meth:`add_fold`.  Subsequent calls
        are no-ops (guarded by an internal ``_finalized`` flag).
        """
        ...

    @property
    def has_fitted_estimators(self) -> bool:
        """Whether fitted estimators are stored in results.

        Returns
        -------
        bool
            ``True`` if at least one fold result contains a non-``None``
            ``fitted_estimator``; ``False`` otherwise or when no folds
            have been added yet.
        """
        if not self.fold_results_:
            return False
        return self.fold_results_[0].fitted_estimator is not None

    def _compute_summary(self, scores_df: pd.DataFrame, label: str = "") -> pd.DataFrame:
        """Compute mean, std, CI, median, IQR for score columns.

        Confidence intervals use the Nadeau--Bengio correction for
        dependent cross-validation folds.

        Parameters
        ----------
        scores_df : pandas.DataFrame
            DataFrame where each row is an outer fold and each column is
            a scoring metric.
        label : str, optional
            Unused label reserved for future use (e.g., distinguishing
            default vs. optimised summaries).

        Returns
        -------
        pandas.DataFrame
            Summary table with columns ``metric``, ``mean``, ``std``,
            ``ci_lower``, ``ci_upper``, ``median``, ``iqr``.

        Notes
        -----
        The corrected confidence interval half-width is computed as:

        .. math::

            h = t_{\\alpha/2,\\,n-1} \\cdot
            \\sqrt{\\frac{1}{n} + \\frac{n_{\\text{test}}}{n_{\\text{train}}}}
            \\cdot s

        where *s* is the sample standard deviation of the per-fold
        scores.  This accounts for the non-independence of folds that
        share training data, following [1]_.

        References
        ----------
        .. [1] Nadeau, C. and Bengio, Y. (2003). "Inference for the
           Generalization Error." *Machine Learning*, 52(3), 239--281.
        """
        n = len(scores_df)
        summary_rows = []
        for col in scores_df.columns:
            vals = scores_df[col].values
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)

            # Nadeau-Bengio corrected CI (average fold sizes for unequal folds)
            if n > 1 and self.fold_results_:
                n_test = np.mean([len(fr.test_indices) for fr in self.fold_results_])
                n_train = np.mean([len(fr.train_indices) for fr in self.fold_results_])
                correction = np.sqrt((1.0 / n) + (n_test / n_train))
                t_crit = t_dist.ppf(0.975, df=n - 1)
                ci_half = t_crit * correction * std
            else:
                ci_half = 0.0

            summary_rows.append(
                {
                    "metric": col,
                    "mean": mean,
                    "std": std,
                    "ci_lower": mean - ci_half,
                    "ci_upper": mean + ci_half,
                    "median": np.median(vals),
                    "iqr": np.subtract(*np.percentile(vals, [75, 25])),
                }
            )
        return pd.DataFrame(summary_rows)

    def to_dict(self) -> dict:
        """Convert results to a plain Python dictionary.

        Returns
        -------
        dict
            Dictionary with keys ``"n_outer_folds"``,
            ``"feature_names"``, and (if available)
            ``"best_params_per_fold"``.

        Examples
        --------
        >>> results.finalize()
        >>> d = results.to_dict()  # doctest: +SKIP
        >>> d["n_outer_folds"]  # doctest: +SKIP
        5
        """
        result = {
            "n_outer_folds": self.n_outer_folds_,
            "feature_names": self.feature_names_in_,
        }
        if hasattr(self, "best_params_per_fold_"):
            result["best_params_per_fold"] = self.best_params_per_fold_
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert per-fold scores to a DataFrame.

        Returns
        -------
        pandas.DataFrame
            A copy of ``outer_scores_default_`` where each row
            corresponds to an outer fold and each column to a scoring
            metric.  Returns an empty DataFrame if ``finalize`` has not
            been called or no default scores are available.

        See Also
        --------
        summary_default_ : Aggregated summary statistics over folds.
        """
        if hasattr(self, "outer_scores_default_"):
            return self.outer_scores_default_.copy()
        return pd.DataFrame()

    def to_json(self, path: str | None = None) -> str:
        """Export results as a JSON string, optionally writing to file.

        NumPy arrays, integers, floats, and DataFrames are automatically
        converted to JSON-serialisable types.

        Parameters
        ----------
        path : str or None, optional
            If given, the JSON string is also written to this file path.

        Returns
        -------
        str
            JSON-formatted string of the results dictionary.

        Examples
        --------
        >>> json_str = results.to_json()  # doctest: +SKIP
        >>> results.to_json("/tmp/results.json")  # doctest: +SKIP
        """
        import json

        data = self.to_dict()

        # Convert numpy types
        def _convert(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            return obj

        json_str = json.dumps(data, default=_convert, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(json_str)
        return json_str

    def to_latex(self) -> str:
        """Export the default summary table as a LaTeX tabular string.

        Returns
        -------
        str
            LaTeX source for the summary table (metric, mean, std, CI
            bounds, median, IQR).  Returns an empty string if
            ``summary_default_`` is not available.

        Examples
        --------
        >>> print(results.to_latex())  # doctest: +SKIP
        \\begin{tabular}{lrrrrrrr}
        ...
        \\end{tabular}
        """
        if hasattr(self, "summary_default_"):
            return self.summary_default_.to_latex(index=False, float_format="%.4f")
        return ""
