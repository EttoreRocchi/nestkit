"""Hyperparameter stability analysis across outer folds.

Provides ``HyperparameterStability``, a diagnostic class for assessing
how consistently the inner cross-validation selects hyperparameters
across outer folds.  Unstable hyperparameter selection may indicate
that the model is sensitive to the training data composition.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd


class HyperparameterStability:
    """Assess hyperparameter selection consistency across outer folds.

    Analyses the best hyperparameter configurations chosen in each
    outer fold and provides summary statistics (mode, entropy,
    agreement rate, coefficient of variation), pairwise Jaccard
    similarity, and a stability flag per parameter.

    Parameters
    ----------
    best_params_per_fold : list of dict
        Best hyperparameters selected in each outer fold.  Each dict
        maps parameter names to their selected values.

    Attributes
    ----------
    best_params_per_fold : list of dict
        The input parameter sets (stored by reference).
    n_folds : int
        Number of outer folds (``len(best_params_per_fold)``).

    Examples
    --------
    >>> from nestkit.diagnostics.stability import HyperparameterStability
    >>> params = [
    ...     {"C": 1.0, "kernel": "rbf"},
    ...     {"C": 1.0, "kernel": "rbf"},
    ...     {"C": 0.1, "kernel": "rbf"},
    ... ]
    >>> hs = HyperparameterStability(params)
    >>> hs.summary()  # doctest: +SKIP
      param mode  nunique   entropy  agreement_rate   cv
    0     C  1.0        2  0.918...        0.666667  ...
    1  kernel  rbf      1  0.000000        1.000000  NaN
    """

    def __init__(self, best_params_per_fold: list[dict]):
        self.best_params_per_fold = best_params_per_fold
        self.n_folds = len(best_params_per_fold)

    def summary(self) -> pd.DataFrame:
        """Compute per-parameter stability summary statistics.

        Returns
        -------
        pandas.DataFrame
            One row per hyperparameter with columns:

            * ``param`` -- Hyperparameter name.
            * ``mode`` -- Most frequently selected value (as string).
            * ``nunique`` -- Number of distinct values across folds.
            * ``entropy`` -- Shannon entropy (base 2) of the value
              distribution.  Zero means perfect agreement.
            * ``agreement_rate`` -- Fraction of folds that selected the
              modal value.  Ranges from ``1/n_folds`` to 1.
            * ``cv`` -- Coefficient of variation (``std / mean``) for
              numeric parameters.  ``NaN`` for non-numeric parameters.

        Notes
        -----
        Values are converted to strings for counting purposes, so
        ``1`` and ``1.0`` are treated as distinct.

        Examples
        --------
        >>> hs = HyperparameterStability([{"lr": 0.01}, {"lr": 0.01}])
        >>> hs.summary()["agreement_rate"].iloc[0]  # doctest: +SKIP
        1.0
        """
        all_params: set[str] = set()
        for bp in self.best_params_per_fold:
            all_params.update(bp.keys())

        rows = []
        for param in sorted(all_params):
            vals = [bp.get(param) for bp in self.best_params_per_fold]
            str_vals = [str(v) for v in vals]
            counts = Counter(str_vals)
            mode_val, mode_count = counts.most_common(1)[0]

            # Entropy
            probs = np.array(list(counts.values())) / self.n_folds
            entropy = float(-np.sum(probs * np.log2(probs + 1e-15)))

            # CV for numeric params
            cv = np.nan
            try:
                numeric_vals = [float(v) for v in vals if v is not None]
                if len(numeric_vals) == self.n_folds and np.mean(numeric_vals) != 0:
                    cv = float(np.std(numeric_vals, ddof=1) / np.mean(numeric_vals))
            except (TypeError, ValueError):
                pass

            rows.append(
                {
                    "param": param,
                    "mode": mode_val,
                    "nunique": len(counts),
                    "entropy": entropy,
                    "agreement_rate": mode_count / self.n_folds,
                    "cv": cv,
                }
            )

        return pd.DataFrame(rows)

    def is_stable(self, threshold: float = 0.8) -> dict[str, bool]:
        """Determine whether each hyperparameter is stable.

        A parameter is considered stable if its agreement rate (fraction
        of folds selecting the modal value) meets or exceeds the given
        threshold.

        Parameters
        ----------
        threshold : float, default 0.8
            Minimum agreement rate to consider a parameter stable.

        Returns
        -------
        dict of {str: bool}
            Mapping from parameter name to stability flag.

        Examples
        --------
        >>> hs = HyperparameterStability([
        ...     {"C": 1.0}, {"C": 1.0}, {"C": 0.1}
        ... ])
        >>> hs.is_stable(threshold=0.5)  # doctest: +SKIP
        {'C': True}
        >>> hs.is_stable(threshold=0.8)  # doctest: +SKIP
        {'C': False}
        """
        df = self.summary()
        return {row["param"]: row["agreement_rate"] >= threshold for _, row in df.iterrows()}

    def pairwise_jaccard(self) -> pd.DataFrame:
        """Compute pairwise Jaccard similarity of hyperparameter configurations.

        Treats each fold's selected configuration as a set of
        ``"param=value"`` strings and computes the Jaccard index for
        every pair of folds.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``fold_i``, ``fold_j``, ``jaccard``.
            One row per unique pair of folds.

        Notes
        -----
        The Jaccard similarity index is defined as:

        .. math::

            J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}

        where *A* and *B* are the sets of ``"param=value"`` strings for
        two folds.  A Jaccard index of 1.0 means the two folds selected
        identical configurations; 0.0 means completely different
        configurations.

        Examples
        --------
        >>> hs = HyperparameterStability([
        ...     {"C": 1.0, "kernel": "rbf"},
        ...     {"C": 1.0, "kernel": "rbf"},
        ...     {"C": 0.1, "kernel": "linear"},
        ... ])
        >>> hs.pairwise_jaccard()  # doctest: +SKIP
           fold_i  fold_j  jaccard
        0       0       1      1.0
        1       0       2      0.0
        2       1       2      0.0
        """
        rows = []
        for i, j in combinations(range(self.n_folds), 2):
            set_i = set(f"{k}={v}" for k, v in self.best_params_per_fold[i].items())
            set_j = set(f"{k}={v}" for k, v in self.best_params_per_fold[j].items())
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            jaccard = intersection / union if union > 0 else 1.0
            rows.append({"fold_i": i, "fold_j": j, "jaccard": jaccard})
        return pd.DataFrame(rows)
