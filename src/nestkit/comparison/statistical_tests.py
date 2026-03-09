"""Statistical hypothesis tests for comparing cross-validated models.

Provides frequentist (corrected paired t-test) and Bayesian (correlated
t-test) procedures, as well as Holm--Bonferroni p-value correction for
multiple comparisons.

References
----------
.. [1] Nadeau, C. and Bengio, Y. (2003). "Inference for the
       Generalization Error." *Machine Learning*, 52(3), 239--281.
.. [2] Benavoli, A., Corani, G., Demsar, J., and Zaffalon, M. (2017).
       "Time for a Change: a Tutorial for Comparing Multiple Classifiers
       Through Bayesian Analysis." *JMLR*, 18(77), 1--36.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as t_dist


def nadeau_bengio_corrected_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_train: int,
    n_test: int,
) -> dict:
    """Nadeau--Bengio corrected paired t-test.

    Standard paired t-tests underestimate variance when applied to
    cross-validation scores because the training sets overlap.  This
    correction inflates the variance estimate by a factor of
    ``(1/k + n_test/n_train)`` where *k* is the number of folds.

    Parameters
    ----------
    scores_a : numpy.ndarray
        Per-fold scores for model A, shape ``(n_folds,)``.
    scores_b : numpy.ndarray
        Per-fold scores for model B, shape ``(n_folds,)``.
    n_train : int
        Number of training samples per fold.
    n_test : int
        Number of test samples per fold.

    Returns
    -------
    dict
        ``t_statistic`` : float
            Corrected t-statistic.
        ``p_value`` : float
            Two-sided p-value from the t-distribution with
            ``n_folds - 1`` degrees of freedom.
        ``mean_difference`` : float
            Mean of ``scores_a - scores_b``.
        ``corrected_std`` : float
            Standard deviation after the Nadeau--Bengio correction.
        ``ci_lower``, ``ci_upper`` : float
            95 % confidence interval for the mean difference.
        ``n_folds`` : int
        ``significant_at_005`` : bool
        ``significant_at_001`` : bool

    Notes
    -----
    The corrected variance is:

    .. math::

        \\sigma^2_{\\text{corr}} =
        \\left(\\frac{1}{k} + \\frac{n_{\\text{test}}}{n_{\\text{train}}}
        \\right) \\hat{\\sigma}^2_d

    where :math:`\\hat{\\sigma}^2_d` is the sample variance of the
    per-fold score differences.

    References
    ----------
    .. [1] Nadeau, C. and Bengio, Y. (2003). *Machine Learning*,
           52(3), 239--281.

    See Also
    --------
    bayesian_correlated_ttest : Bayesian alternative.
    holm_bonferroni_correction : Multiple-comparison adjustment.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([0.90, 0.88, 0.91, 0.87, 0.89])
    >>> b = np.array([0.85, 0.84, 0.86, 0.83, 0.85])
    >>> result = nadeau_bengio_corrected_ttest(a, b, 800, 200)
    >>> result["significant_at_005"]  # doctest: +SKIP
    True
    """
    diffs = scores_a - scores_b
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    var_diff = float(np.var(diffs, ddof=1))

    # Nadeau-Bengio correction
    correction = (1.0 / n) + (n_test / n_train)
    corrected_var = correction * var_diff
    corrected_std = float(np.sqrt(corrected_var))

    if corrected_std == 0:
        # When all fold differences are identical: if nonzero, evidence
        # against H_0 is maximal; if zero, no difference exists.
        if mean_diff != 0:
            return {
                "t_statistic": float(np.sign(mean_diff) * np.inf),
                "p_value": 0.0,
                "mean_difference": mean_diff,
                "corrected_std": 0.0,
                "ci_lower": mean_diff,
                "ci_upper": mean_diff,
                "n_folds": n,
                "significant_at_005": True,
                "significant_at_001": True,
            }
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "mean_difference": mean_diff,
            "corrected_std": 0.0,
            "ci_lower": mean_diff,
            "ci_upper": mean_diff,
            "n_folds": n,
            "significant_at_005": False,
            "significant_at_001": False,
        }

    t_stat = mean_diff / corrected_std
    p_value = float(2 * t_dist.sf(abs(t_stat), df=n - 1))

    t_crit = float(t_dist.ppf(0.975, df=n - 1))
    ci_lower = mean_diff - t_crit * corrected_std
    ci_upper = mean_diff + t_crit * corrected_std

    return {
        "t_statistic": float(t_stat),
        "p_value": p_value,
        "mean_difference": mean_diff,
        "corrected_std": corrected_std,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_folds": n,
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
    }


def bayesian_correlated_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_train: int,
    n_test: int,
    rope: float = 0.01,
) -> dict:
    """Bayesian correlated t-test for model comparison.

    Uses a Student-t posterior distribution over the mean score
    difference and partitions probability mass into three regions:
    model A is better, the models are practically equivalent (within
    the ROPE), or model B is better.  The standard error is inflated
    by the Nadeau--Bengio correction to account for the
    non-independence of cross-validation fold scores.

    Parameters
    ----------
    scores_a : numpy.ndarray
        Per-fold scores for model A, shape ``(n_folds,)``.
    scores_b : numpy.ndarray
        Per-fold scores for model B, shape ``(n_folds,)``.
    n_train : int
        Number of training samples per fold.
    n_test : int
        Number of test samples per fold.
    rope : float, default=0.01
        Half-width of the Region of Practical Equivalence.  Differences
        in ``[-rope, +rope]`` are considered practically meaningless.

    Returns
    -------
    dict
        ``p_a_better`` : float
            Posterior probability that model A is better (difference >
            *rope*).
        ``p_equivalent`` : float
            Posterior probability that models are practically equivalent.
        ``p_b_better`` : float
            Posterior probability that model B is better (difference <
            -*rope*).
        ``rope`` : float
            The ROPE value used.
        ``mean_difference`` : float
            Point estimate of the mean score difference.
        ``hdi_lower``, ``hdi_upper`` : float
            95 % highest-density interval of the mean difference.

    Notes
    -----
    The standard error uses the Nadeau--Bengio correction:

    .. math::

        \\text{SE} = s_d \\cdot
        \\sqrt{\\frac{1}{k} + \\frac{n_{\\text{test}}}{n_{\\text{train}}}}

    The three probabilities always sum to 1 (up to floating-point
    rounding).  When *rope* = 0 the test reduces to a standard
    Bayesian sign test.

    References
    ----------
    .. [1] Benavoli, A., Corani, G., Demsar, J., and Zaffalon, M.
           (2017). "Time for a Change: a Tutorial for Comparing
           Multiple Classifiers Through Bayesian Analysis." *JMLR*,
           18(77), 1--36.
    .. [2] Nadeau, C. and Bengio, Y. (2003). *Machine Learning*,
           52(3), 239--281.

    See Also
    --------
    nadeau_bengio_corrected_ttest : Frequentist alternative.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([0.90, 0.88, 0.91, 0.87, 0.89])
    >>> b = np.array([0.85, 0.84, 0.86, 0.83, 0.85])
    >>> result = bayesian_correlated_ttest(a, b, n_train=800, n_test=200, rope=0.01)
    >>> result["p_a_better"] > 0.5  # doctest: +SKIP
    True
    """
    diffs = scores_a - scores_b
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    if std_diff == 0 or n < 2:
        if abs(mean_diff) <= rope:
            return {
                "p_a_better": 0.0,
                "p_equivalent": 1.0,
                "p_b_better": 0.0,
                "rope": rope,
                "mean_difference": mean_diff,
                "hdi_lower": mean_diff,
                "hdi_upper": mean_diff,
            }
        return {
            "p_a_better": 1.0 if mean_diff > 0 else 0.0,
            "p_equivalent": 0.0,
            "p_b_better": 0.0 if mean_diff > 0 else 1.0,
            "rope": rope,
            "mean_difference": mean_diff,
            "hdi_lower": mean_diff,
            "hdi_upper": mean_diff,
        }

    # Nadeau-Bengio corrected standard error
    correction = (1.0 / n) + (n_test / n_train)
    se = std_diff * np.sqrt(correction)
    df = n - 1

    # Probabilities
    p_a_better = float(1 - t_dist.cdf(rope, df=df, loc=mean_diff, scale=se))
    p_b_better = float(t_dist.cdf(-rope, df=df, loc=mean_diff, scale=se))
    p_equivalent = 1 - p_a_better - p_b_better

    # 95% HDI
    t_crit = float(t_dist.ppf(0.975, df=df))
    hdi_lower = mean_diff - t_crit * se
    hdi_upper = mean_diff + t_crit * se

    return {
        "p_a_better": max(0, p_a_better),
        "p_equivalent": max(0, p_equivalent),
        "p_b_better": max(0, p_b_better),
        "rope": rope,
        "mean_difference": mean_diff,
        "hdi_lower": float(hdi_lower),
        "hdi_upper": float(hdi_upper),
    }


def holm_bonferroni_correction(p_values: list[float]) -> list[float]:
    """Apply the Holm--Bonferroni step-down correction to p-values.

    Controls the family-wise error rate (FWER) when performing multiple
    pairwise comparisons.  Less conservative than the classical
    Bonferroni correction because it uses a step-down procedure.

    Parameters
    ----------
    p_values : list[float]
        Uncorrected p-values, one per comparison.

    Returns
    -------
    list[float]
        Corrected p-values in the same order as the input.  Each value
        is clipped to ``[0, 1]`` and monotonicity is enforced so that
        the corrected p-value of a less significant test is never
        smaller than that of a more significant one.

    Notes
    -----
    For the *i*-th smallest p-value (0-indexed), the corrected value is
    ``min(1, p_i * (m - i))`` where *m* is the total number of tests.

    References
    ----------
    .. [1] Holm, S. (1979). "A Simple Sequentially Rejective Multiple
           Test Procedure." *Scandinavian Journal of Statistics*, 6(2),
           65--70.

    See Also
    --------
    nadeau_bengio_corrected_ttest : Produces the raw p-values.

    Examples
    --------
    >>> holm_bonferroni_correction([0.01, 0.04, 0.03])
    [0.03, 0.04, 0.06]
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    corrected = np.zeros(n)

    for rank, idx in enumerate(sorted_indices):
        corrected[idx] = min(1.0, p_values[idx] * (n - rank))

    # Enforce monotonicity
    for i in range(1, n):
        idx = sorted_indices[i]
        prev_idx = sorted_indices[i - 1]
        corrected[idx] = max(corrected[idx], corrected[prev_idx])

    return corrected.tolist()
