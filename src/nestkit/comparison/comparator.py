"""Statistical model comparison for nested cross-validation results.

This module provides :class:`NestedCVComparator`, which performs
statistically rigorous pairwise and multi-model comparisons using
corrected paired t-tests, Bayesian tests, and Holm--Bonferroni
multiple-comparison correction.

References
----------
.. [1] Nadeau, C. and Bengio, Y. (2003). "Inference for the
       Generalization Error." *Machine Learning*, 52(3), 239--281.
.. [2] Benavoli, A., Corani, G., Demsar, J., and Zaffalon, M. (2017).
       "Time for a Change: a Tutorial for Comparing Multiple Classifiers
       Through Bayesian Analysis." *JMLR*, 18(77), 1--36.
.. [3] Demsar, J. (2006). "Statistical Comparisons of Classifiers over
       Multiple Data Sets." *JMLR*, 7, 1--30.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

from nestkit.comparison.statistical_tests import (
    bayesian_correlated_ttest,
    holm_bonferroni_correction,
    nadeau_bengio_corrected_ttest,
)
from nestkit.results._base import _BaseNestedCVResults


class NestedCVComparator:
    """Statistically rigorous comparison of nested cross-validation results.

    Provides corrected paired t-tests (Nadeau & Bengio, 2003), Bayesian
    correlated t-tests (Benavoli et al., 2017), and Holm--Bonferroni
    multiple-comparison correction (Demsar, 2006) for comparing two or
    more models that were evaluated on **identical** outer folds.

    All registered models must share the same outer-fold split indices;
    this is validated automatically when a new model is added.

    Attributes
    ----------
    _results : dict[str, _BaseNestedCVResults]
        Mapping from model name to its nested CV results object.

    Examples
    --------
    >>> comparator = NestedCVComparator()  # doctest: +SKIP
    >>> comparator.add("rf", rf_results)  # doctest: +SKIP
    >>> comparator.add("svm", svm_results)  # doctest: +SKIP
    >>> comparator.summary("accuracy")  # doctest: +SKIP

    See Also
    --------
    nestkit.comparison.statistical_tests.nadeau_bengio_corrected_ttest
    nestkit.comparison.statistical_tests.bayesian_correlated_ttest

    References
    ----------
    .. [1] Nadeau, C. and Bengio, Y. (2003). "Inference for the
           Generalization Error." *Machine Learning*, 52(3), 239--281.
    .. [2] Benavoli, A. et al. (2017). *JMLR*, 18(77), 1--36.
    .. [3] Demsar, J. (2006). *JMLR*, 7, 1--30.
    """

    def __init__(self):
        self._results: dict[str, _BaseNestedCVResults] = {}

    def add(self, name: str, results: _BaseNestedCVResults) -> None:
        """Register a model's nested cross-validation results.

        Parameters
        ----------
        name : str
            Unique human-readable identifier for the model (e.g.,
            ``"random_forest"``).
        results : _BaseNestedCVResults
            Fitted nested CV results object.  Must contain per-fold
            ``test_indices`` that match every previously registered model.

        Raises
        ------
        ValueError
            If the outer-fold structure of *results* does not match that
            of models already registered.

        See Also
        --------
        _validate_fold_alignment : Alignment check called internally.
        """
        if self._results:
            self._validate_fold_alignment(name, results)
        self._results[name] = results

    def _validate_fold_alignment(self, name: str, new_results: _BaseNestedCVResults) -> None:
        """Verify that outer-fold indices match exactly.

        Compares the number of outer folds and the test-set indices of
        every fold against the first registered model.

        Parameters
        ----------
        name : str
            Name of the model being added.
        new_results : _BaseNestedCVResults
            Nested CV results to validate.

        Raises
        ------
        ValueError
            If the number of outer folds or any fold's test indices
            differ from the reference model.
        """
        ref_name, ref_results = next(iter(self._results.items()))

        if new_results.n_outer_folds_ != ref_results.n_outer_folds_:
            raise ValueError(
                f"Model '{name}' has {new_results.n_outer_folds_} outer folds, "
                f"but '{ref_name}' has {ref_results.n_outer_folds_}. "
                f"All models must use identical outer_cv splits."
            )

        for k in range(ref_results.n_outer_folds_):
            ref_test = ref_results.fold_results_[k].test_indices
            new_test = new_results.fold_results_[k].test_indices
            if not np.array_equal(ref_test, new_test):
                raise ValueError(
                    f"Outer fold {k}: test indices for '{name}' do not match "
                    f"'{ref_name}'. All models must be evaluated on identical "
                    f"outer folds for valid comparison."
                )

    def _get_scores(self, model: str, metric: str, threshold: str = "default") -> np.ndarray:
        """Extract per-fold scores for a registered model.

        Parameters
        ----------
        model : str
            Name of the registered model.
        metric : str
            Scoring metric key (e.g., ``"accuracy"``, ``"roc_auc"``).
        threshold : {"default", "optimized"}, default="default"
            Whether to use default or threshold-optimized outer scores.

        Returns
        -------
        numpy.ndarray
            1-D array of shape ``(n_outer_folds,)`` with per-fold scores.

        Raises
        ------
        ValueError
            If *threshold* is ``"optimized"`` but the model has no
            optimized scores.
        KeyError
            If *model* was not previously registered via :meth:`add`.
        """
        results = self._results[model]
        if threshold == "optimized":
            if not hasattr(results, "outer_scores_optimized_"):
                raise ValueError(f"Model '{model}' has no optimized scores.")
            return results.outer_scores_optimized_[metric].values
        return results.outer_scores_default_[metric].values

    def summary(self, metric: str, threshold: str = "default") -> pd.DataFrame:
        """Produce a side-by-side summary table of all registered models.

        For each model the table includes the mean, standard deviation,
        median, 95 % confidence interval (Nadeau--Bengio corrected,
        t-distribution), min, max, and inter-quartile range of the
        outer-fold scores.

        Parameters
        ----------
        metric : str
            Scoring metric key to summarise.
        threshold : {"default", "optimized"}, default="default"
            Which score variant to use.

        Returns
        -------
        pandas.DataFrame
            One row per model with columns ``model``, ``mean``, ``std``,
            ``median``, ``ci_lower``, ``ci_upper``, ``min``, ``max``,
            ``iqr``.

        Examples
        --------
        >>> comparator.summary("roc_auc")  # doctest: +SKIP
        """
        rows = []
        for name in self._results:
            scores = self._get_scores(name, metric, threshold)
            mean = float(np.mean(scores))
            std = float(np.std(scores, ddof=1))
            n = len(scores)

            # Nadeau-Bengio corrected CI using t-distribution
            results_obj = self._results[name]
            n_test = np.mean([len(fr.test_indices) for fr in results_obj.fold_results_])
            n_train = np.mean([len(fr.train_indices) for fr in results_obj.fold_results_])
            correction = np.sqrt((1.0 / n) + (n_test / n_train))
            t_crit = float(t_dist.ppf(0.975, df=n - 1))
            ci_half = t_crit * correction * std

            rows.append(
                {
                    "model": name,
                    "mean": mean,
                    "std": std,
                    "median": float(np.median(scores)),
                    "ci_lower": mean - ci_half,
                    "ci_upper": mean + ci_half,
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "iqr": float(np.subtract(*np.percentile(scores, [75, 25]))),
                }
            )
        return pd.DataFrame(rows)

    def corrected_paired_ttest(
        self, metric: str, model_a: str, model_b: str, threshold: str = "default"
    ) -> dict:
        """Perform the Nadeau--Bengio corrected paired t-test.

        Accounts for the non-independence of cross-validation fold
        scores caused by overlapping training sets.

        Parameters
        ----------
        metric : str
            Scoring metric key.
        model_a : str
            Name of the first model.
        model_b : str
            Name of the second model.
        threshold : {"default", "optimized"}, default="default"
            Which score variant to use.

        Returns
        -------
        dict
            Test results including ``t_statistic``, ``p_value``,
            ``mean_difference``, ``corrected_std``, ``ci_lower``,
            ``ci_upper``, ``n_folds``, ``significant_at_005``,
            ``significant_at_001``.

        See Also
        --------
        nestkit.comparison.statistical_tests.nadeau_bengio_corrected_ttest

        References
        ----------
        .. [1] Nadeau, C. and Bengio, Y. (2003). *Machine Learning*,
               52(3), 239--281.
        """
        scores_a = self._get_scores(model_a, metric, threshold)
        scores_b = self._get_scores(model_b, metric, threshold)
        folds = self._results[model_a].fold_results_
        n_test = np.mean([len(fr.test_indices) for fr in folds])
        n_train = np.mean([len(fr.train_indices) for fr in folds])
        return nadeau_bengio_corrected_ttest(scores_a, scores_b, n_train, n_test)

    def pairwise_corrected_ttest(self, metric: str, threshold: str = "default") -> pd.DataFrame:
        """Run corrected paired t-tests for every model pair.

        All ``C(n, 2)`` pairwise Nadeau--Bengio tests are performed and
        then adjusted for multiple comparisons via the step-down
        Holm--Bonferroni procedure.

        Parameters
        ----------
        metric : str
            Scoring metric key.
        threshold : {"default", "optimized"}, default="default"
            Which score variant to use.

        Returns
        -------
        pandas.DataFrame
            One row per pair with columns ``model_a``, ``model_b``,
            all keys from :func:`nadeau_bengio_corrected_ttest`, and
            ``p_value_corrected``.

        See Also
        --------
        corrected_paired_ttest : Single-pair test.
        nestkit.comparison.statistical_tests.holm_bonferroni_correction

        References
        ----------
        .. [1] Demsar, J. (2006). *JMLR*, 7, 1--30.
        """
        models = list(self._results.keys())
        rows = []
        p_values = []

        for a, b in combinations(models, 2):
            result = self.corrected_paired_ttest(metric, a, b, threshold)
            rows.append({"model_a": a, "model_b": b, **result})
            p_values.append(result["p_value"])

        if rows:
            corrected = holm_bonferroni_correction(p_values)
            for i, row in enumerate(rows):
                row["p_value_corrected"] = corrected[i]

        return pd.DataFrame(rows)

    def bayesian_comparison(
        self,
        metric: str,
        model_a: str,
        model_b: str,
        rope: float = 0.01,
        threshold: str = "default",
    ) -> dict:
        """Perform a Bayesian correlated t-test between two models.

        Uses a Student-t posterior over the mean score difference and
        partitions the probability mass into three regions: model A
        better, practically equivalent (within the ROPE), and model B
        better.

        Parameters
        ----------
        metric : str
            Scoring metric key.
        model_a : str
            Name of the first model.
        model_b : str
            Name of the second model.
        rope : float, default=0.01
            Half-width of the Region of Practical Equivalence.
        threshold : {"default", "optimized"}, default="default"
            Which score variant to use.

        Returns
        -------
        dict
            Posterior probabilities and diagnostics: ``p_a_better``,
            ``p_equivalent``, ``p_b_better``, ``rope``,
            ``mean_difference``, ``hdi_lower``, ``hdi_upper``.

        See Also
        --------
        nestkit.comparison.statistical_tests.bayesian_correlated_ttest

        References
        ----------
        .. [1] Benavoli, A. et al. (2017). *JMLR*, 18(77), 1--36.
        """
        scores_a = self._get_scores(model_a, metric, threshold)
        scores_b = self._get_scores(model_b, metric, threshold)
        folds = self._results[model_a].fold_results_
        n_test = np.mean([len(fr.test_indices) for fr in folds])
        n_train = np.mean([len(fr.train_indices) for fr in folds])
        return bayesian_correlated_ttest(scores_a, scores_b, n_train, n_test, rope)

    def rank_models(self, metric: str, threshold: str = "default") -> pd.DataFrame:
        """Rank all registered models by mean outer-fold score.

        Returns the same summary table as :meth:`summary` sorted in
        descending order of ``mean`` with an additional ``rank`` column
        (1 = best).

        Parameters
        ----------
        metric : str
            Scoring metric key.
        threshold : {"default", "optimized"}, default="default"
            Which score variant to use.

        Returns
        -------
        pandas.DataFrame
            Sorted summary table with an extra ``rank`` column.

        See Also
        --------
        summary : Unsorted summary table.
        """
        summary = self.summary(metric, threshold)
        summary = summary.sort_values("mean", ascending=False).reset_index(drop=True)
        summary["rank"] = range(1, len(summary) + 1)
        return summary
